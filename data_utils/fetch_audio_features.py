"""
Fetches audio features for songs listed in data/to_fetch.csv.

Pipeline per song:
  1. Deezer Search API   -> deezer_id
  2. Soundcharts API     -> audio features by deezer_id

Writes JSONL shards to data/fetched/shards/shard_XXXXXX.jsonl
Resumable: already-written shards are skipped on re-run.
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ---------- CONFIG ----------
TOTAL = 15000          # сколько песен из to_fetch.csv обработать (начиная с начала)
BATCH_SIZE = 100      # сколько песен в одном шарде
DEEZER_DELAY = 0.0     # без паузы — Deezer держит ~50 req/sec
SOUNDCHARTS_DELAY = 0.0
HTTP_TIMEOUT = 10

SC_APP_ID = "TSMITH55-API_FD41E748"
SC_API_KEY = "529a3a2d40b6a49d"   # TODO: ротируй ключ и читай из env

IN_CSV = Path("data/to_fetch.csv")
OUT_DIR = Path("data/fetched/shards")
MISSING_CSV = Path("data/fetched/missing.csv")
# ----------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
MISSING_CSV.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch")


def clean_title(t: str) -> str:
    """Убираем (Album Version), [Remastered], feat. X и т.п."""
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", t)
    t = re.sub(r"\s*feat\..*|\s*ft\..*|\s*-\s*Remaster.*", "", t, flags=re.I)
    return t.strip()


def deezer_search(artist: str, title: str) -> int | None:
    """Строгий поиск -> при промахе нестрогий. Возвращает Deezer track id."""
    attempts = [
        {"q": f'artist:"{artist}" track:"{title}"'},
        {"q": f'artist:"{artist}" track:"{clean_title(title)}"'},
        {"q": f"{artist} {clean_title(title)}"},
    ]
    for params in attempts:
        params["limit"] = 1
        try:
            r = requests.get(
                "https://api.deezer.com/search", params=params, timeout=HTTP_TIMEOUT
            )
            data = r.json().get("data", [])
            if data:
                return data[0]["id"]
        except Exception as e:
            log.warning(f"deezer error: {e}")
        time.sleep(DEEZER_DELAY)
    return None


def soundcharts_features(deezer_id: int) -> dict | None:
    url = f"https://customer.api.soundcharts.com/api/v2.25/song/by-platform/deezer/{deezer_id}"
    headers = {"x-app-id": SC_APP_ID, "x-api-key": SC_API_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"soundcharts error for deezer={deezer_id}: {e}")
        return None


def main():
    df = pd.read_csv(IN_CSV).head(TOTAL)
    log.info(f"Loaded {len(df)} songs from {IN_CSV}")

    total_processed = 0
    t0 = time.time()

    for start in range(0, len(df), BATCH_SIZE):
        shard_path = OUT_DIR / f"shard_{start:06d}.jsonl"
        batch = df.iloc[start : start + BATCH_SIZE]

        if shard_path.exists():
            log.info(f"[{start}/{len(df)}] shard exists, skip: {shard_path.name}")
            total_processed += len(batch)
            continue

        log.info(f"[{start}/{len(df)}] starting shard {shard_path.name} ({len(batch)} songs)")
        rows = []

        for i, (_, row) in enumerate(batch.iterrows(), 1):
            global_idx = start + i
            log.info(f"  ({global_idx}/{len(df)}) {row.artist!r} — {row.title!r}")

            deezer_id = deezer_search(row.artist, row.title)
            if deezer_id is None:
                rows.append({"song_id": row.song_id, "status": "deezer_not_found"})
                continue

            features = soundcharts_features(deezer_id)
            if features is None:
                rows.append({"song_id": row.song_id, "deezer_id": deezer_id,
                             "status": "soundcharts_not_found"})
            else:
                rows.append({"song_id": row.song_id, "deezer_id": deezer_id,
                             "status": "ok", "features": features})

            time.sleep(SOUNDCHARTS_DELAY)

        shard_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        total_processed += len(batch)

        ok = sum(1 for r in rows if r.get("status") == "ok")
        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed else 0
        eta = (len(df) - total_processed) / rate if rate else 0
        log.info(
            f"[{total_processed}/{len(df)}] shard done: {ok}/{len(rows)} ok | "
            f"rate={rate:.1f} song/s | eta={eta/60:.1f} min"
        )

    # Собираем missing из всех шардов (включая ранее сохранённые)
    all_rows = []
    for f in sorted(OUT_DIR.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))

    missing_rows = [r for r in all_rows if r.get("status") != "ok"]
    if missing_rows:
        meta = pd.read_csv(IN_CSV).set_index("song_id")[["artist", "title"]]
        miss_df = pd.DataFrame(missing_rows).join(meta, on="song_id")
        miss_df.to_csv(MISSING_CSV, index=False)
        log.info(f"missing: {len(miss_df)} songs -> {MISSING_CSV}")

    log.info(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()