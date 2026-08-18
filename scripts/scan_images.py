import chromadb, json
from collections import defaultdict

client = chromadb.PersistentClient(path="./data/chroma")
col = client.get_collection("obsidian_notes")

# pull everything (ids + metadatas). Chroma caps get() at some limit; loop if needed.
all_ids = col.get(limit=col.count())["ids"]
print("total chunks:", len(all_ids))

# Fetch in batches to be safe
B = 5000
metas = []
for i in range(0, len(all_ids), B):
    batch = col.get(ids=all_ids[i:i+B], include=["metadatas"])["metadatas"]
    metas.extend(batch)

image_by_file = defaultdict(lambda: {"with_asset": [], "no_asset": [], "kinds": set()})
total_image = 0
for m in metas:
    if not m:
        continue
    kind = (m.get("kind") or "")
    asset = m.get("asset_id") or None
    # image chunk detection: kind == image, OR asset_id present, OR img_url present
    is_img = (kind == "image") or (asset is not None) or bool(m.get("img_url"))
    if is_img:
        total_image += 1
        fp = m.get("filepath") or m.get("source") or "?"
        rec = image_by_file[fp]
        rec["kinds"].add(kind)
        if asset:
            rec["with_asset"].append(asset)
        else:
            rec["no_asset"].append(m.get("heading_path") or m.get("title") or "?")

print("total image chunks:", total_image)
print("notes with image chunks:", len(image_by_file))
print("=" * 70)
for fp in sorted(image_by_file):
    rec = image_by_file[fp]
    wa = len(rec["with_asset"])
    na = len(rec["no_asset"])
    print(f"\n[{fp}]")
    print(f"  kind tags: {sorted(rec['kinds'])}")
    print(f"  with asset_id (DISPLAYABLE): {wa}")
    print(f"  missing asset_id (no display): {na}")
    if rec["with_asset"]:
        print(f"  asset_ids: {rec['with_asset']}")
