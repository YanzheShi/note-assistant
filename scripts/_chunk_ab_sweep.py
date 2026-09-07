#!/usr/bin/env python3
"""
切分参数扫跑驱动：逐列「全量重建索引 → 只跑方式 A 评测」，保证两件事严格同序：
索引产物（Chroma / BM25 / WikiGraph / v2b docstore）与评测进程读到的是同一套切分参数。

列与列之间只差索引参数，故评测差异可归因到切分本身。

用法：
    # 跑一列
    python scripts/_chunk_ab_sweep.py --arms v2b_500_100:v2b:500:100

    # 跑多列（顺序即执行顺序；末列建议留生产默认，跑完索引就停在默认状态）
    python scripts/_chunk_ab_sweep.py \
        --arms v2_500_100:v2:500:100 v2_800_150:v2:800:150 v2b_800_150:v2b:800:150

    # 已有报告的列跳过（断点续跑）
    python scripts/_chunk_ab_sweep.py --arms ... --skip-existing

每列产出到 eval/report/<name>/：reindex.log、_meta.json（索引侧统计）、
eval.log、rerank_ab_A.json。之后用 scripts/_compare_chunk_ab.py 做多列对比。

前置：Ollama 起且 bge-m3 可用（重建索引要本地 embedding）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = ROOT / "eval" / "report"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arm(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 4:
        raise SystemExit(f"--arms 需为 NAME:STRATEGY:CHUNK_SIZE:CHUNK_OVERLAP，收到：{spec}")
    name, strategy, size, overlap = parts
    if strategy not in ("v1", "v2", "v2b"):
        raise SystemExit(f"未知 chunking_strategy：{strategy}")
    return {"name": name, "strategy": strategy, "size": int(size), "overlap": int(overlap)}


def index_is_current(out_dir: Path, arm: dict, expected_vault: str | None = None) -> bool:
    """本列索引是否已是目标切分参数下的成品（用于断点续跑，省一次全量重建）。

    同时要求参数一致、vault 一致与「落盘 chunk 数 == Chroma 实际计数」：最后一项能识别出
    重建中途被杀导致的半成品索引，此时必须重跑而不是复用。
    """
    mp = out_dir / "_meta.json"
    if not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if (meta.get("chunk_size"), meta.get("chunk_overlap"), meta.get("chunking_strategy")) != \
            (arm["size"], arm["overlap"], arm["strategy"]):
        return False
    if expected_vault and str(meta.get("vault", "")).rstrip("\\/") != expected_vault.rstrip("\\/"):
        print(f"  [index] vault 不符（{meta.get('vault')} ≠ {expected_vault}）→ 重建")
        return False
    expected = meta.get("chunks")
    if not expected:
        return False
    try:
        import chromadb
        sys.path.insert(0, str(ROOT / "src"))
        from note_assistant.config import settings
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir.resolve()))
        live = client.get_collection(settings.collection_name).count()
    except Exception as e:  # noqa: BLE001
        print(f"  [index] 无法核对当前索引计数（视为需重建）: {e}")
        return False
    print(f"  [index] live={live} meta.chunks={expected} → "
          f"{'复用' if live == expected else '参数不符，重建'}")
    return live == expected


def run(cmd: list[str], env: dict, log_path: Path, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"$ {' '.join(cmd)}\n$ env: VAULT_PATH={env.get('VAULT_PATH')} "
                 f"CHUNKING_STRATEGY={env.get('CHUNKING_STRATEGY')} "
                 f"CHUNK_SIZE={env.get('CHUNK_SIZE')} CHUNK_OVERLAP={env.get('CHUNK_OVERLAP')}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"  [{label}] exit={proc.returncode} → {log_path.relative_to(ROOT)}")
    return proc.returncode


def write_meta(out_dir: Path, arm: dict) -> dict:
    """从 reindex.log 抽 chunks / 总耗时，连同本列参数落 _meta.json 供对比工具读取。"""
    log = (out_dir / "reindex.log").read_text(encoding="utf-8", errors="replace")
    chunks = re.search(r"Chunks:\s*(\d+)", log)
    secs = re.search(r"全流程总耗时:\s*([\d.]+)s", log)
    files = re.search(r"文件数:\s*(\d+)", log)
    vault = re.search(r"Vault:\s*(.+)", log)
    meta = {
        "chunking_strategy": arm["strategy"],
        "chunk_size": arm["size"],
        "chunk_overlap": arm["overlap"],
        "vault": vault.group(1).strip() if vault else None,
        "files": int(files.group(1)) if files else None,
        "chunks": int(chunks.group(1)) if chunks else None,
        "reindex_seconds": float(secs.group(1)) if secs else None,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [meta] {meta}")
    return meta


def contaminated_questions(report: Path) -> list[int]:
    """返回「答案为空」的题号——429 限流被图内 try/except 吞掉时的典型指纹。"""
    if not report.exists():
        return [-1]
    try:
        d = json.loads(report.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [-1]
    return [i for i, q in enumerate(d.get("per_question") or [], 1)
            if not (q.get("generated_answer") or "").strip()]


def main():
    ap = argparse.ArgumentParser(description="切分参数扫跑（重建索引 + 方式 A 评测）")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="NAME:STRATEGY:CHUNK_SIZE:CHUNK_OVERLAP，可多个，按序执行")
    ap.add_argument("--vault", default=None,
                    help="评测用 vault 绝对路径；缺省沿用 .env 的 VAULT_PATH。"
                         "只按进程环境变量注入，不改 .env（避免影响本地 API 服务）")
    ap.add_argument("--subset", type=int, default=10, help="评测题数（默认全 10 题）")
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--skip-existing", action="store_true", help="已有 rerank_ab_A.json 的列跳过")
    ap.add_argument("--force-reindex", action="store_true", help="即使盘上已有同参数索引也强制重建")
    ap.add_argument("--question-gap", type=float, default=0.0,
                    help="题间冷却秒数（注入 EVAL_QUESTION_GAP_SECONDS），免费额度按 RPM 限流时用")
    ap.add_argument("--arm-cooldown", type=float, default=0.0,
                    help="列与列之间的冷却秒数，让限流窗口恢复")
    args = ap.parse_args()

    if args.vault:
        # config.py 里 load_dotenv(".env.local", override=True) 会盖过进程环境变量，
        # 若 .env.local 恰好钉死了 VAULT_PATH，本列会静默跑在错误的库上。
        local = ROOT / ".env.local"
        if local.exists() and re.search(r"(?im)^\s*VAULT_PATH\s*=", local.read_text(encoding="utf-8", errors="replace")):
            raise SystemExit(
                "!! .env.local 用 override=True 钉死了 VAULT_PATH，--vault 会被它静默覆盖。\n"
                "   请临时注释掉该行，或直接改 .env 的 VAULT_PATH 后去掉 --vault。"
            )

    for spec in args.arms:
        arm = parse_arm(spec)
        out_dir = REPORT_ROOT / arm["name"]
        report = out_dir / "rerank_ab_A.json"
        print("=" * 70)
        print(f"列 {arm['name']}: strategy={arm['strategy']} chunk_size={arm['size']} "
              f"chunk_overlap={arm['overlap']}")
        print("=" * 70)
        if args.skip_existing and report.exists():
            print("  已有报告，跳过")
            continue

        env = dict(os.environ)
        if args.vault:
            env["VAULT_PATH"] = str(Path(args.vault).resolve())
        env["CHUNKING_STRATEGY"] = arm["strategy"]
        env["CHUNK_SIZE"] = str(arm["size"])
        env["CHUNK_OVERLAP"] = str(arm["overlap"])
        env["PYTHONUNBUFFERED"] = "1"
        if args.question_gap:
            env["EVAL_QUESTION_GAP_SECONDS"] = str(args.question_gap)

        t0 = time.time()
        if args.force_reindex or not index_is_current(out_dir, arm, env.get("VAULT_PATH")):
            rc = run([sys.executable, "scripts/full_reindex.py", "--yes"], env, out_dir / "reindex.log", "reindex")
            if rc != 0:
                print(f"  !! 重建失败，跳过本列评测（详见 {out_dir / 'reindex.log'}）")
                continue
            write_meta(out_dir, arm)
            print(f"  reindex wall={time.time() - t0:.0f}s")
        else:
            print(f"  索引已是本列参数成品，跳过重建（--force-reindex 可强制）")

        # 索引刚落盘，稍退避再进评测，避免 agnes 免费额度连续打点
        time.sleep(5)
        rc = run([
            sys.executable, "scripts/run_eval_rerank_ab.py",
            "--variants", "A", "--subset", str(args.subset),
            "--k", *[str(k) for k in args.k], "-o", str(out_dir),
        ], env, out_dir / "eval.log", "eval")
        if rc != 0 or not report.exists():
            print(f"  !! 评测异常，报告可能缺失（详见 {out_dir / 'eval.log'}）")
            continue

        bad = contaminated_questions(report)
        if bad:
            for f in (report, out_dir / "rerank_ab_compare.json",
                      out_dir / "rerank_ab_compare.md", out_dir / "rerank_ab_compare.html"):
                f.unlink(missing_ok=True)
            print(f"  !! 本列被限流污染（{len(bad)}/{args.subset} 题空答案），报告已弃用。")
            print("     后续列会在同一限流窗口内跑出同样的垃圾数据，扫跑中止。")
            print("     等额度窗口恢复后加 --skip-existing 续跑，或调大 --question-gap。")
            sys.exit(2)

        print(f"  本列干净：{args.subset}/{args.subset} 题均有答案")
        time.sleep(args.arm_cooldown)
        print()


if __name__ == "__main__":
    main()
