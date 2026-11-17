"""Fetch official open data, validate publisher hashes, and safely extract PNGs."""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download(row):
    target = ROOT / "data/raw" / row["filename"]
    expected = row["content_details"]["sha256_hash"]
    if target.exists():
        h = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if h != expected:
            raise ValueError(f"Existing archive hash mismatch: {target.name}")
        return target
    part = target.with_suffix(".zip.partial")
    req = urllib.request.Request(row["content_details"]["download_url"], headers={"User-Agent": "Mozilla/5.0"})
    h = hashlib.sha256(); total = 0; last = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as response, part.open("xb") as out:
        while block := response.read(4 * 1024 * 1024):
            out.write(block); h.update(block); total += len(block)
            if time.monotonic() - last > 15:
                print(json.dumps({"file": target.name, "downloaded_mb": round(total / 1e6), "total_mb": round(row["size"] / 1e6)}), flush=True)
                last = time.monotonic()
    if total != row["size"] or h.hexdigest() != expected:
        raise ValueError(f"Publisher size/hash verification failed: {target.name}")
    part.rename(target)
    print(json.dumps({"verified": target.name, "sha256": expected}), flush=True)
    return target


def main():
    if shutil.disk_usage(ROOT).free < 13 * 1024**3:
        raise RuntimeError("At least 13 GiB free disk is required")
    rows = json.loads((ROOT / "docs/sources/dataset-image-files.json").read_text())
    image_dir = ROOT / "data/images"; image_dir.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(download, rows))
    count = 0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.lower().endswith(".png"):
                    continue
                name = Path(entry.filename).name
                if name in ("", ".", "..") or entry.file_size > 64 * 1024**2:
                    raise ValueError("Invalid image entry")
                target = image_dir / name
                with archive.open(entry) as src:
                    data = src.read()
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f"Conflicting duplicate filename: {name}")
                target.write_bytes(data); count += 1
    print(json.dumps({"extracted_pngs": count, "unique_pngs": len(list(image_dir.glob('*.png')))}), flush=True)


if __name__ == "__main__":
    main()
