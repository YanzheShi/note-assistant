# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**Obsidian RAG** — a personal knowledge base Q&A system. Ingests an Obsidian vault (Markdown notes with YAML front matter and `[[wikilinks]]`), chunks and embeds them into ChromaDB, then answers natural language questions via a RAG pipeline (hybrid dense+sparse retrieval → reranking → LLM generation). Designed for local-first deployment (embedding + reranker run on-prem, LLM via API).

## Commands

```bash
# Install/sync dependencies (uses uv)
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/indexing/test_vault_loader.py

# Run a single test by name
uv run pytest tests/indexing/test_vault_loader.py::test_scan_excludes_hidden_dirs

# Run the splitter CLI (loads vault, compares v1 vs v2 chunking)
uv run python -m note_assistant.indexing.splitter

# Run full indexing pipeline (requires Ollama running)
uv run python -m note_assistant.indexing.ingestor

# Lint
uv run ruff check .
```

## Architecture

The package lives under `src/note_assistant/`. The codebase is in **early stages** — the `indexing` module is functional; `retrieval`, `generation`, `pipeline`, `api`, and `evaluation` are mostly stubs awaiting implementation.

### Data Flow (indexing)

```
VaultLoader → DocNode → RichPreprocessor → split_v2 → restore → Ingestor → ChromaDB
```

1. **`indexing/vault_loader.py`** — Scans the vault, parses YAML front matter (fault-tolerant: bad FM is skipped, title falls back to filename), extracts `[[wikilinks]]` (deduped, order-preserved), and the heading tree. Returns `DocNode` objects (defined in `indexing/types.py`).
2. **`indexing/preprocessor.py`** — `RichPreprocessor` extracts rich structures (code fences, tables, mermaid, images) into placeholders so the splitter doesn't mangle them. After splitting, `restore()` puts originals back. Also generates "summary chunks" for each extracted structure so they're searchable.
3. **`indexing/splitter.py`** — Two-layer strategy: `MarkdownHeaderTextSplitter` (preserves `#`/`##`/`###`/`####` hierarchy) → `RecursiveCharacterTextSplitter` (800 char / 150 overlap, Chinese-aware with `。` in separators). Produces chunks with `heading_path` metadata like `"一、背景 > 检索方法"`. v1 (flat Recursive) and v3 (per-chunk wikilinks) are stubbed for comparison.
4. **`indexing/embedder.py`** — Wraps Ollama's `bge-m3:latest` model (1024-dim dense vectors).
5. **`indexing/ingestor.py`** — `Ingestor.index_vault()` ties it all together: load → preprocess → split → restore → enrich with wikilinks/metadata → upsert into ChromaDB (cosine similarity).

### Key Design Decisions (see DECISIONS.md for full reasoning)

- **Loader is file-level only** — chunking is the splitter's job. This avoids duplicating heading parsing.
- **Front matter fault tolerance** — never modifies user notes; bad YAML degrades gracefully to empty FM.
- **Two-layer splitter** — preserves heading context that flat Recursive would lose (e.g., a chunk about "FA2 improvements" retains its parent "FlashAttention" h1 heading).
- **ChromaDB over FAISS** — auto-persistence + `where` metadata filtering (by tags/filepath/heading) is valuable for Obsidian workflows.
- **Dual .env loading** — `load_dotenv(PROJECT_ROOT / ".env")` + pydantic-settings `env_file` ensures .env is found regardless of cwd (PyCharm, Docker, `uv run`).
- **Wikilinks are per-document, not per-chunk** (Day 1) — every chunk from a note gets the same wikilinks list. Chunk-level is a Day 3 upgrade.

### Configuration

All config is in `src/note_assistant/config.py` via `pydantic-settings`. Reads `.env` and `.env.local` (override). Key settings: `vault_path`, `ollama_base_url`, `embed_model`, `chroma_persist_dir`, `deepseek_api_key`, `chunk_size`, `bm25_weight`/`dense_weight`, `top_k_retrieve`, `top_k_rerank`.

### What's a Stub vs. What's Real

| Module | Status |
|---|---|
| `indexing/vault_loader.py` | ✅ Functional |
| `indexing/preprocessor.py` | ✅ Functional |
| `indexing/splitter.py` | ✅ v1 + v2 implemented; v2b/v3 stubbed |
| `indexing/embedder.py` | ✅ Functional (requires Ollama) |
| `indexing/ingestor.py` | ✅ Functional (requires Ollama + vault) |
| `retrieval/` | 🚧 Stub only |
| `generation/` | 🚧 Stub only |
| `pipeline/` | 🚧 Stub only |
| `api/` | 🚧 Stub only |
| `evaluation/` | 🚧 Stub only |
| `frontend/app.py` | 🚧 Placeholder (`st.title('Obsidian RAG')`) |

## Testing

- Framework: `pytest` + `pytest-asyncio`
- Test location: `tests/` (mirrors `src/` package structure)
- Tests use `tmp_path` fixtures to create mini vaults — **no dependency on the real vault** for unit tests.
- `tests/test_config.py` uses `monkeypatch.setenv` and `_env_file=None` to isolate from real `.env`.

## Conventions

- Python 3.12+, `pathlib.Path` everywhere, `dataclass` for business types.
- `DocNode` and `ExtractedChunk` (in `indexing/types.py`) are the core business types — keep them framework-agnostic.
- LangChain types (`Document`) are used only at internal boundaries (splitter, vector store); never in function signatures exposed to callers.
- Chinese text is the primary language in the vault — separators and tokenization must be Chinese-aware (`。` in splitter separators, etc.).
