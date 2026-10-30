"""Fetch the frozen H77 BTCUSDT hourly source from Binance public data."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_btc_weekend_monday_v1"
ZIP_DIR = OUT / "source_zip"
BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"


def months() -> list[str]:
    return [f"{year}-{month:02d}" for year in range(2021, 2027)
            for month in range(1, 13) if (year, month) <= (2026, 4)]


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research archive verifier)"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Cannot download {url}: {last}")


def fetch(month: str) -> dict:
    name = f"BTCUSDT-1h-{month}.zip"
    path = ZIP_DIR / name
    url = f"{BASE}/{name}"
    expected_text = get_bytes(url + ".CHECKSUM").decode("utf-8").strip()
    expected = expected_text.split()[0].lower()
    if len(expected) != 64 or not all(c in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid published checksum for {name}: {expected_text!r}")
    payload = path.read_bytes() if path.exists() else get_bytes(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise ValueError(f"Published checksum mismatch for {name}: {digest} != {expected}")
    if not path.exists():
        temp = path.with_suffix(".zip.part")
        temp.write_bytes(payload)
        temp.replace(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError(f"Invalid archive shape: {name}")
        rows = sum(1 for _ in archive.open(names[0]))
    return {
        "month": month, "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "url": url, "sha256": digest, "published_sha256": expected,
        "bytes": len(payload), "rows": rows, "status": "OK",
    }


def main() -> None:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "source_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Source manifest already exists: {manifest}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(fetch, month): month for month in months()}
        for future in concurrent.futures.as_completed(future_map):
            month = future_map[future]
            try:
                record = future.result()
                print(month, record["rows"], record["sha256"][:12], flush=True)
            except Exception as exc:
                record = {"month": month, "status": "ERROR", "error": str(exc)}
                print(month, "ERROR", exc, flush=True)
            results.append(record)
    results.sort(key=lambda x: x["month"])
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Binance public spot BTCUSDT monthly 1h klines",
        "schema": "https://github.com/binance/binance-public-data",
        "records": results,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ok = sum(r["status"] == "OK" for r in results)
    print("ARCHIVES", ok, "/", len(results), "ROWS", sum(r.get("rows", 0) for r in results), flush=True)
    if ok != len(results):
        raise RuntimeError("Source acquisition incomplete; Phase 1 must not run")


if __name__ == "__main__":
    main()
