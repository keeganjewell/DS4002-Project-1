"""
Resume collect_dataset.py from the saved checkpoint instead of re-running
pool-building and the first LRCLIB pass. Loads music_genre_dataset_raw.csv
(the full 1707-song candidate pool) and music_genre_dataset_checkpoint.csv
(lyrics results for the first 1700 of those), merges them, retries the
still-missing lyrics, then continues with language detection and the
final per-genre trim -- same as the tail end of collect_dataset.py.
"""

import pandas as pd
import time
import os

from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 4002
RANDOM_SEED = 4002

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")

SONGS_PER_ARTIST = 5
ARTISTS_PER_GENRE_TARGET = 20

GENRE_TAGS_ORDER = ["Country", "Pop", "Rock", "Hip-Hop/Rap"]

LRCLIB_URL = "https://lrclib.net/api/search"
LRCLIB_HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}

import requests


def get_lrclib_lyrics(song_title, artist_name, max_attempts=3):
    params = {"track_name": song_title, "artist_name": artist_name}
    response = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(LRCLIB_URL, params=params, headers=LRCLIB_HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"    LRCLIB request error: {e}, retrying...")
            time.sleep(3)
            continue
        if response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 5))
            print(f"    rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        break
    if response is None or response.status_code != 200:
        return None, None, None
    results = response.json()
    if not results:
        return None, None, None
    result = results[0]
    return result.get("artistName"), result.get("trackName"), result.get("plainLyrics")


def detect_language(text):
    if pd.isna(text) or not str(text).strip():
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def main():
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "music_genre_dataset_raw.csv"))
    checkpoint = pd.read_csv(os.path.join(DATA_DIR, "music_genre_dataset_checkpoint.csv"))
    print(f"Loaded raw pool: {len(raw_df)} songs")
    print(f"Loaded checkpoint: {len(checkpoint)} songs with lyrics attempted")

    lyrics_cols = checkpoint[["recording_id", "lrclib_artist", "lrclib_title", "lyrics"]]
    raw_df = raw_df.merge(lyrics_cols, on="recording_id", how="left")
    raw_df["lyrics_found"] = raw_df["lyrics"].notna()
    print(f"Lyrics found (from checkpoint): {raw_df['lyrics_found'].sum()} / {len(raw_df)}")

    missing_idx = raw_df[~raw_df["lyrics_found"]].index
    print(f"\nRetrying {len(missing_idx)} missing lyrics...")
    for n, i in enumerate(missing_idx):
        artist = raw_df.loc[i, "artist"]
        title = raw_df.loc[i, "song_title"]
        try:
            matched_artist, matched_title, lyrics = get_lrclib_lyrics(title, artist)
        except Exception as e:
            print(f"    unexpected error retrying '{title}': {e}")
            matched_artist, matched_title, lyrics = None, None, None
        if lyrics is not None:
            raw_df.loc[i, "lrclib_artist"] = matched_artist
            raw_df.loc[i, "lrclib_title"] = matched_title
            raw_df.loc[i, "lyrics"] = lyrics
        if (n + 1) % 25 == 0:
            print(f"  retried {n + 1}/{len(missing_idx)}")
        time.sleep(1.5)

    raw_df["lyrics_found"] = raw_df["lyrics"].notna()
    print(f"Lyrics found after retry: {raw_df['lyrics_found'].sum()} / {len(raw_df)}")

    with_lyrics_path = os.path.join(DATA_DIR, "music_genre_dataset_with_lyrics.csv")
    raw_df.to_csv(with_lyrics_path, index=False)
    print(f"Saved to {with_lyrics_path}")

    print("\n=== Detecting language ===")
    raw_df["language"] = raw_df["lyrics"].apply(detect_language)
    print(raw_df["language"].value_counts(dropna=False))

    clean_df = raw_df[(raw_df["lyrics_found"]) & (raw_df["language"] == "en")].copy()
    print(f"\nEnglish + lyrics-found songs: {len(clean_df)}")
    print(clean_df["target_genre"].value_counts())

    print("\n=== Trimming to target artists/songs per genre ===")
    final_parts = []
    for genre in GENRE_TAGS_ORDER:
        genre_clean = clean_df[clean_df["target_genre"] == genre]
        artist_song_counts = genre_clean.groupby("artist").size()
        full_artists = artist_song_counts[artist_song_counts >= SONGS_PER_ARTIST].index

        n_take = min(ARTISTS_PER_GENRE_TARGET, len(full_artists))
        if n_take < ARTISTS_PER_GENRE_TARGET:
            print(f"  {genre}: only {n_take} artists have {SONGS_PER_ARTIST} full English songs "
                  f"(wanted {ARTISTS_PER_GENRE_TARGET})")

        chosen_artists = pd.Series(full_artists).sample(n=n_take, random_state=RANDOM_SEED)
        genre_final = (
            genre_clean[genre_clean["artist"].isin(chosen_artists)]
            .groupby("artist", group_keys=False)
            .sample(n=SONGS_PER_ARTIST, random_state=RANDOM_SEED)
        )
        final_parts.append(genre_final)
        print(f"  {genre}: final {genre_final['artist'].nunique()} artists, {len(genre_final)} songs")

    final_df = pd.concat(final_parts, ignore_index=True)
    final_path = os.path.join(DATA_DIR, "music_genre_dataset_final.csv")
    final_df.to_csv(final_path, index=False)

    print(f"\n=== FINAL DATASET: {len(final_df)} songs ===")
    print(final_df["target_genre"].value_counts())
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
