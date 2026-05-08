"""
Собирает из шардов data/fetched/shards/ два финальных датасета:

1) data/fetched/audio_features.parquet
   song_id + плоские аудио-признаки (по одной строке на песню)

2) data/interactions_audio_v2.parquet
   user-song взаимодействия для песен, у которых есть фичи
"""

import json
from pathlib import Path

import pandas as pd

SHARDS_DIR = Path("data/fetched/shards")
FEATURES_OUT = Path("data/fetched/audio_features_v2.1.parquet")
INTERACTIONS_IN = Path("data/interactions_full.parquet")
INTERACTIONS_OUT = Path("data/interactions_audio_v2.1.parquet")


def load_shards():
    rows = []
    for f in sorted(SHARDS_DIR.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_shards()
    print(f"total rows in shards: {len(rows)}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"ok rows: {len(ok_rows)}")

    # смотрим структуру features одного ответа, чтобы понять, что разворачивать
    sample = ok_rows[0]["features"]
    print("sample features keys:", list(sample.keys())[:10])

    # Soundcharts возвращает JSON типа {"object": {...}} или {"audio": {...}}
    # подстрой под реальную структуру после прогона sample
    flat = []
    for r in ok_rows:
        obj = r["features"].get("object", {})
        audio = obj.get("audio") or {}
        if not audio:
            continue
        artists = [a.get("name") for a in (obj.get("mainArtists") or obj.get("artists") or [])]
        genres_root = [g.get("root") for g in (obj.get("genres") or []) if g.get("root")]
        genres_sub = [s for g in (obj.get("genres") or []) for s in (g.get("sub") or [])]
        flat.append({
            "song_id": r["song_id"],
            "deezer_id": r.get("deezer_id"),
            "name": obj.get("name"),
            "artists": ", ".join(artists) if artists else None,
            "genre_root": ", ".join(genres_root) if genres_root else None,
            "genre_sub": ", ".join(genres_sub) if genres_sub else None,
            "duration": obj.get("duration"),
            "release_date": obj.get("releaseDate"),
            "explicit": obj.get("explicit"),
            "language_code": obj.get("languageCode"),
            **audio,
        })

    features_df = pd.DataFrame(flat).drop_duplicates("song_id")
    print(f"unique songs with features: {len(features_df)}")
    print("columns:", list(features_df.columns))

    features_df.to_parquet(FEATURES_OUT, index=False)
    print(f"saved {FEATURES_OUT}")

    # Сборка interactions_audio_v2
    users = pd.read_parquet(INTERACTIONS_IN)
    inter = users.merge(features_df, on="song_id", how="inner")
    print(f"interactions with audio: {len(inter):,} rows, "
          f"{inter['user_id'].nunique()} users, {inter['song_id'].nunique()} songs")
    inter.to_parquet(INTERACTIONS_OUT, index=False)
    print(f"saved {INTERACTIONS_OUT}")


if __name__ == "__main__":
    main()