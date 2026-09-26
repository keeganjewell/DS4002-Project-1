"""
PIPELINE STEP 5 of 6: Country class-balance backfill (raw data -> final data).

Must run AFTER 04_collect_dataset.py, using its output as the starting
point. The original collection run (step 04) only found 15 candidate
Country artists (single "country" tag search), and after the popularity /
lyrics / language filters only 13 survived with 5 full English songs --
short of the 20-artist target that Pop, Rock, and Hip-Hop/Rap all hit.

This script widens the Country tag search (mirroring the approach already
used for Pop and Hip-Hop/Rap in 04_collect_dataset.py), pulls a larger
candidate pool, excludes artists already in the dataset, and runs the same
popularity -> lyrics -> language pipeline to backfill new Country artists
up to the 20-artist / 100-song target, so all four genres end up balanced
at 100 songs / 20 artists each.

Run with: python SCRIPTS/05_backfill_country.py
Requires: DATA/music_genre_dataset_{raw,with_lyrics,final}.csv to already
          exist (i.e. 04_collect_dataset.py has been run); internet access
          (MusicBrainz, ListenBrainz, LRCLIB APIs); takes on the order of
          tens of minutes due to API rate limiting.
Output: overwrites DATA/music_genre_dataset_raw.csv,
        DATA/music_genre_dataset_with_lyrics.csv,
        DATA/music_genre_dataset_final.csv
with the existing rows plus the new Country backfill merged in. After this
step, music_genre_dataset_final.csv is the final, class-balanced modeling
dataset (400 songs, 100/genre, 20 artists/genre).
"""

import requests
import pandas as pd
import time
import os

from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 4002
RANDOM_SEED = 4002

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")

HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}
MB_BASE_URL = "https://musicbrainz.org/ws/2/recording"
LB_URL = "https://api.listenbrainz.org/1/popularity/artist"
LRCLIB_URL = "https://lrclib.net/api/search"
LRCLIB_HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}

MIN_LISTENERS = 1000
SONGS_PER_ARTIST = 5
SONGS_PER_ARTIST_CANDIDATES = 8
ARTISTS_PER_GENRE_TARGET = 20
CANDIDATE_PAGES = 5

# Widened Country tag variants, same pattern used for Pop/Hip-Hop/Rap
COUNTRY_TAGS_WIDE = [
    "country", "country rock", "country pop", "alternative country",
    "contemporary country", "classic country", "outlaw country",
    "americana", "bluegrass", "country blues",
]


# NOTE: the helper functions below (get_musicbrainz_data through
# detect_language) are the same MusicBrainz/ListenBrainz/LRCLIB/language-
# detection helpers as in 04_collect_dataset.py -- see that script's
# comments for a detailed explanation of each. They're duplicated here
# (rather than imported) so this script can run standalone. The only new
# logic specific to this backfill step is in main() below.

def get_musicbrainz_data(params, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            response = requests.get(MB_BASE_URL, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"    request error: {e}, retrying...")
            time.sleep(3)
            continue
        if response.status_code == 200:
            data = response.json()
            if "recordings" in data:
                return data
        elif response.status_code == 503:
            print(f"    MusicBrainz busy, waiting 5s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(5)
        else:
            print(f"    unexpected status {response.status_code}: {response.text[:200]}")
            break
    raise Exception("MusicBrainz request failed after multiple attempts.")


def fetch_candidates(search_tag, pages=CANDIDATE_PAGES):
    all_recordings = []
    for page in range(pages):
        offset = page * 100
        if offset + 100 > 500:
            break
        params = {"query": f'tag:"{search_tag}"', "fmt": "json", "limit": 100, "offset": offset}
        data = get_musicbrainz_data(params)
        recs = data.get("recordings", [])
        if not recs:
            break
        all_recordings.extend(recs)
        time.sleep(1.1)
    return all_recordings


def recordings_to_df(recordings, target_genre):
    records = []
    for r in recordings:
        artist_names, artist_ids = [], []
        for credit in r.get("artist-credit", []):
            if isinstance(credit, dict):
                artist_names.append(credit.get("name", ""))
                artist_ids.append(credit.get("artist", {}).get("id", ""))
        tags = [t["name"] for t in r.get("tags", [])]
        records.append({
            "recording_id": r.get("id"),
            "song_title": r.get("title"),
            "artist": ", ".join(artist_names),
            "artist_id": ", ".join(artist_ids),
            "musicbrainz_tags": ", ".join(tags),
            "target_genre": target_genre,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.drop_duplicates(subset="recording_id").reset_index(drop=True)


def get_listenbrainz_popularity(artist_mbid, max_attempts=3):
    payload = {"artist_mbids": [artist_mbid]}
    for attempt in range(max_attempts):
        try:
            response = requests.post(LB_URL, json=payload, timeout=30)
        except requests.RequestException:
            time.sleep(2)
            continue
        if response.status_code == 429:
            time.sleep(3)
            continue
        if response.status_code != 200:
            return None, None
        result = response.json()
        if not result:
            return None, None
        return result[0].get("total_listen_count"), result[0].get("total_user_count")
    return None, None


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
    # Load the outputs already produced by 04_collect_dataset.py -- this
    # script only ADDS new Country artists to them, it doesn't start over.
    final_df = pd.read_csv(os.path.join(DATA_DIR, "music_genre_dataset_final.csv"))
    raw_df_existing = pd.read_csv(os.path.join(DATA_DIR, "music_genre_dataset_raw.csv"))
    wl_df_existing = pd.read_csv(os.path.join(DATA_DIR, "music_genre_dataset_with_lyrics.csv"))

    existing_country_artists = set(final_df[final_df["target_genre"] == "Country"]["artist"])
    n_have = len(existing_country_artists)
    n_need = ARTISTS_PER_GENRE_TARGET - n_have
    print(f"Existing Country artists in final dataset: {n_have}")
    print(f"Need {n_need} more artists to hit {ARTISTS_PER_GENRE_TARGET}")

    if n_need <= 0:
        print("Already at target, nothing to do.")
        return

    # Exclude artists already seen in the raw candidate pool (whether they
    # ended up in the final dataset or not) so this backfill doesn't waste
    # API calls re-querying lyrics for artists we've already tried.
    already_tried_artists = set(wl_df_existing[wl_df_existing["target_genre"] == "Country"]["artist"])

    print(f"\n=== Building widened Country candidate pool (tags: {COUNTRY_TAGS_WIDE}) ===")
    all_recordings = []
    for tag in COUNTRY_TAGS_WIDE:
        recordings = fetch_candidates(tag)
        print(f"  '{tag}': {len(recordings)} raw recordings")
        all_recordings.extend(recordings)
        time.sleep(1.5)

    df = recordings_to_df(all_recordings, "Country")
    print(f"  Unique recordings: {len(df)}")

    # Exclude recording_ids already in our existing pool, and exclude
    # artists we've already tried (whether they succeeded or failed)
    df = df[~df["artist"].isin(already_tried_artists)].copy()
    print(f"  Recordings from NEW artists only: {len(df)}")

    artist_counts = df["artist"].value_counts()
    eligible = artist_counts[artist_counts >= SONGS_PER_ARTIST]

    artist_df = (
        df[df["artist"].isin(eligible.index)][["artist", "artist_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    artist_df = artist_df[~artist_df["artist_id"].str.contains(",", na=False)].copy()
    artist_df = artist_df[artist_df["artist_id"] != ""].copy()
    print(f"  New artists with >={SONGS_PER_ARTIST} songs, single-credit: {len(artist_df)}")

    listen_counts, user_counts = [], []
    for i, artist_id in enumerate(artist_df["artist_id"]):
        listens, users = get_listenbrainz_popularity(artist_id)
        listen_counts.append(listens)
        user_counts.append(users)
        time.sleep(0.5)
        if (i + 1) % 25 == 0:
            print(f"    ListenBrainz checked {i + 1}/{len(artist_df)}")

    artist_df["listen_count"] = listen_counts
    artist_df["user_count"] = user_counts

    popular = artist_df[artist_df["user_count"].fillna(0) >= MIN_LISTENERS].copy()
    print(f"  New artists with >={MIN_LISTENERS:,} listeners: {len(popular)}")

    if popular.empty:
        print("  No new eligible artists found. Consider widening tags further.")
        return

    # Pull candidate songs (extra per artist, for lyric/language attrition slack)
    selection = (
        df[df["artist"].isin(popular["artist"])]
        .groupby("artist", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), SONGS_PER_ARTIST_CANDIDATES), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    print(f"  Candidate songs pulled from {selection['artist'].nunique()} new artists: {len(selection)}")

    # ---- Lyrics ----
    print("\n=== Fetching lyrics from LRCLIB for new Country candidates ===")
    matched_artists, matched_titles, lyrics_list = [], [], []
    for i, row in selection.iterrows():
        try:
            matched_artist, matched_title, lyrics = get_lrclib_lyrics(row["song_title"], row["artist"])
        except Exception as e:
            print(f"    unexpected error on '{row['song_title']}': {e}")
            matched_artist, matched_title, lyrics = None, None, None
        matched_artists.append(matched_artist)
        matched_titles.append(matched_title)
        lyrics_list.append(lyrics)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(selection)} processed")
        time.sleep(1)

    selection["lrclib_artist"] = matched_artists
    selection["lrclib_title"] = matched_titles
    selection["lyrics"] = lyrics_list
    selection["lyrics_found"] = selection["lyrics"].notna()
    print(f"Lyrics found: {selection['lyrics_found'].sum()} / {len(selection)}")

    # Retry pass
    missing_idx = selection[~selection["lyrics_found"]].index
    print(f"\nRetrying {len(missing_idx)} missing lyrics...")
    for i in missing_idx:
        artist = selection.loc[i, "artist"]
        title = selection.loc[i, "song_title"]
        try:
            matched_artist, matched_title, lyrics = get_lrclib_lyrics(title, artist)
        except Exception as e:
            print(f"    unexpected error retrying '{title}': {e}")
            matched_artist, matched_title, lyrics = None, None, None
        if lyrics is not None:
            selection.loc[i, "lrclib_artist"] = matched_artist
            selection.loc[i, "lrclib_title"] = matched_title
            selection.loc[i, "lyrics"] = lyrics
        time.sleep(1.5)

    selection["lyrics_found"] = selection["lyrics"].notna()
    print(f"Lyrics found after retry: {selection['lyrics_found'].sum()} / {len(selection)}")

    # ---- Language filter ----
    print("\n=== Detecting language ===")
    selection["language"] = selection["lyrics"].apply(detect_language)
    print(selection["language"].value_counts(dropna=False))

    clean = selection[(selection["lyrics_found"]) & (selection["language"] == "en")].copy()
    print(f"\nNew Country songs, English + lyrics-found: {len(clean)}")

    # ---- Merge the new Country candidates into the existing raw /
    # with_lyrics datasets (append + de-dup by recording_id, so re-running
    # this script is safe and won't create duplicate rows) ----
    raw_new_cols = ["recording_id", "song_title", "artist", "artist_id", "musicbrainz_tags", "target_genre"]
    raw_df_combined = pd.concat(
        [raw_df_existing, selection[raw_new_cols]], ignore_index=True
    ).drop_duplicates(subset="recording_id").reset_index(drop=True)

    wl_cols = raw_new_cols + ["lrclib_artist", "lrclib_title", "lyrics", "lyrics_found", "language"]
    wl_df_combined = pd.concat(
        [wl_df_existing, selection[wl_cols]], ignore_index=True
    ).drop_duplicates(subset="recording_id").reset_index(drop=True)

    raw_path = os.path.join(DATA_DIR, "music_genre_dataset_raw.csv")
    wl_path = os.path.join(DATA_DIR, "music_genre_dataset_with_lyrics.csv")
    raw_df_combined.to_csv(raw_path, index=False)
    wl_df_combined.to_csv(wl_path, index=False)
    print(f"\nUpdated raw pool ({len(raw_df_combined)} songs) saved to {raw_path}")
    print(f"Updated with-lyrics pool ({len(wl_df_combined)} songs) saved to {wl_path}")

    # ---- Rebuild final dataset: Country up to 20 artists x 5 songs, others
    # left exactly as they were (Pop/Rock/Hip-Hop already hit their target
    # in step 04, so nothing about them needs to change here) ----
    print("\n=== Rebuilding final dataset ===")
    other_genres_final = final_df[final_df["target_genre"] != "Country"]

    country_pool = wl_df_combined[
        (wl_df_combined["target_genre"] == "Country")
        & (wl_df_combined["lyrics_found"])
        & (wl_df_combined["language"] == "en")
    ]
    artist_song_counts = country_pool.groupby("artist").size()
    full_artists = artist_song_counts[artist_song_counts >= SONGS_PER_ARTIST].index
    print(f"  Country: {len(full_artists)} total artists with >={SONGS_PER_ARTIST} full English songs available")

    n_take = min(ARTISTS_PER_GENRE_TARGET, len(full_artists))
    chosen_artists = pd.Series(full_artists).sample(n=n_take, random_state=RANDOM_SEED)
    country_final = (
        country_pool[country_pool["artist"].isin(chosen_artists)]
        .groupby("artist", group_keys=False)
        .sample(n=SONGS_PER_ARTIST, random_state=RANDOM_SEED)
    )
    print(f"  Country final: {country_final['artist'].nunique()} artists, {len(country_final)} songs")

    final_combined = pd.concat([other_genres_final, country_final], ignore_index=True)
    final_path = os.path.join(DATA_DIR, "music_genre_dataset_final.csv")
    final_combined.to_csv(final_path, index=False)

    print(f"\n=== FINAL DATASET: {len(final_combined)} songs ===")
    print(final_combined["target_genre"].value_counts())
    print(final_combined.groupby("target_genre")["artist"].nunique())
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
