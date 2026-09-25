"""
Full dataset collection pipeline for the lyrics/genre classification project.

For each of Country, Pop, Rock, Hip-Hop/Rap:
  1. Query MusicBrainz (broadened tag variants for Pop/Rock/Hip-Hop/Rap,
     single tag for Country -- see scale_check.py for why) for candidate
     recordings.
  2. Filter to artists with >= SONGS_PER_ARTIST_CANDIDATES songs, single-
     artist credit only (no collaborations), and >= MIN_LISTENERS
     ListenBrainz listeners.
  3. Pull up to SONGS_PER_ARTIST_CANDIDATES songs from EVERY eligible
     artist (not a capped subset) -- this is the full backfill reservoir.
  4. Fetch lyrics for every candidate song from LRCLIB, with a retry pass.
  5. Detect language and keep English-only, non-missing lyrics.
  6. Per genre, keep artists with >= SONGS_PER_ARTIST surviving English
     songs, randomly select ARTISTS_PER_GENRE_TARGET of them (seed=4002),
     and sample exactly SONGS_PER_ARTIST songs from each.

Target: 400 songs total (100/genre, 20 artists x 5 songs/genre), matching
the project's Goal Statement and preprocessing plan. Pulling extra
candidate songs per artist (8 instead of 5) and drawing from the FULL
eligible-artist pool (not a fixed oversample count) means a genre with
higher non-English/lyrics-missing rates (e.g. Rock's gothic/darkwave
subculture skewing toward German/Russian lyrics) still has enough
surviving artists to reach 20 without a shortfall.

Output: DATA/music_genre_dataset_raw.csv (all selected songs, pre-lyrics)
        DATA/music_genre_dataset_with_lyrics.csv (post-LRCLIB, pre-language-filter)
        DATA/music_genre_dataset_final.csv (English-only, lyrics-found -- the modeling dataset)
"""

import requests
import pandas as pd
import time
import os

from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 4002
RANDOM_SEED = 4002

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}
MB_BASE_URL = "https://musicbrainz.org/ws/2/recording"
LB_URL = "https://api.listenbrainz.org/1/popularity/artist"
LRCLIB_URL = "https://lrclib.net/api/search"
LRCLIB_HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}

MIN_LISTENERS = 1000
SONGS_PER_ARTIST = 5
SONGS_PER_ARTIST_CANDIDATES = 8  # pull extra songs per artist so a lyric/language miss or two doesn't sink the whole artist
ARTISTS_PER_GENRE_TARGET = 20  # final target per genre
CANDIDATE_PAGES = 5  # 5 x 100 = 500 raw recordings per tag (MusicBrainz caps offset+limit at 500)

GENRE_TAGS = {
    "Country": [
        "country",
        "country rock",
        "contemporary country",
        "alternative country",
        "outlaw country"
    ],
    "Pop": ["pop", "pop rock", "dance-pop", "electropop", "synthpop"],
    "Rock": ["rock", "classic rock", "alternative rock", "hard rock", "indie rock"],
    "Hip-Hop/Rap": ["hip hop", "rap", "hip-hop", "gangsta rap", "trap"],
}


# --------------------------------------------------------------------------
# MusicBrainz
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# ListenBrainz
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# LRCLIB
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Per-genre artist pool building + selection
# --------------------------------------------------------------------------

def build_artist_pool(search_tags, target_genre):
    print(f"\n=== Building candidate pool: {target_genre} (tags: {search_tags}) ===")

    all_recordings = []
    for tag in search_tags:
        recordings = fetch_candidates(tag)
        print(f"  '{tag}': {len(recordings)} raw recordings")
        all_recordings.extend(recordings)
        time.sleep(1.5)

    df = recordings_to_df(all_recordings, target_genre)
    print(f"  Unique recordings: {len(df)}")
    if df.empty:
        return df, None

    # Eligibility uses the minimum needed (SONGS_PER_ARTIST); the larger
    # SONGS_PER_ARTIST_CANDIDATES only controls how many songs we *pull*
    # per artist below, so it doesn't shrink the eligible-artist pool.
    artist_counts = df["artist"].value_counts()
    eligible = artist_counts[artist_counts >= SONGS_PER_ARTIST]

    artist_df = (
        df[df["artist"].isin(eligible.index)][["artist", "artist_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    artist_df = artist_df[~artist_df["artist_id"].str.contains(",", na=False)].copy()
    artist_df = artist_df[artist_df["artist_id"] != ""].copy()
    print(f"  Artists with >={SONGS_PER_ARTIST} songs, single-credit: {len(artist_df)}")

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
    print(f"  Artists with >={MIN_LISTENERS:,} listeners: {len(popular)}")

    return df, popular


def select_candidate_songs(df, popular_artists):
    """Take up to SONGS_PER_ARTIST_CANDIDATES songs from every eligible,
    popular artist (not just a capped subset), so there's a full backfill
    reservoir to draw from after lyrics/language attrition."""
    selection = (
        df[df["artist"].isin(popular_artists["artist"])]
        .groupby("artist", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), SONGS_PER_ARTIST_CANDIDATES), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    return selection


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    all_selections = []
    pools = {}

    for genre, tags in GENRE_TAGS.items():
        df, popular = build_artist_pool(tags, genre)
        pools[genre] = (df, popular)

        if popular is None or popular.empty:
            print(f"  WARNING: no eligible artists for {genre}, skipping.")
            continue

        selection = select_candidate_songs(df, popular)
        print(f"  Selected {selection['artist'].nunique()} artists, "
              f"{len(selection)} candidate songs for {genre}")
        all_selections.append(selection)
        time.sleep(2)

    raw_df = pd.concat(all_selections, ignore_index=True)
    raw_path = os.path.join(DATA_DIR, "music_genre_dataset_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw selection ({len(raw_df)} songs) to {raw_path}")
    print(raw_df["target_genre"].value_counts())

    # ---- Lyrics ----
    print("\n=== Fetching lyrics from LRCLIB ===")
    matched_artists, matched_titles, lyrics_list = [], [], []
    for i, row in raw_df.iterrows():
        try:
            matched_artist, matched_title, lyrics = get_lrclib_lyrics(row["song_title"], row["artist"])
        except Exception as e:
            print(f"    unexpected error on '{row['song_title']}': {e}")
            matched_artist, matched_title, lyrics = None, None, None
        matched_artists.append(matched_artist)
        matched_titles.append(matched_title)
        lyrics_list.append(lyrics)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(raw_df)} processed")
        if (i + 1) % 100 == 0:
            checkpoint = raw_df.iloc[:i + 1].copy()
            checkpoint["lrclib_artist"] = matched_artists
            checkpoint["lrclib_title"] = matched_titles
            checkpoint["lyrics"] = lyrics_list
            checkpoint.to_csv(os.path.join(DATA_DIR, "music_genre_dataset_checkpoint.csv"), index=False)
        time.sleep(1)

    raw_df["lrclib_artist"] = matched_artists
    raw_df["lrclib_title"] = matched_titles
    raw_df["lyrics"] = lyrics_list
    raw_df["lyrics_found"] = raw_df["lyrics"].notna()

    print(f"Lyrics found: {raw_df['lyrics_found'].sum()} / {len(raw_df)}")

    # One retry pass for misses
    missing_idx = raw_df[~raw_df["lyrics_found"]].index
    print(f"\nRetrying {len(missing_idx)} missing lyrics...")
    for i in missing_idx:
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
        time.sleep(1.5)

    raw_df["lyrics_found"] = raw_df["lyrics"].notna()
    print(f"Lyrics found after retry: {raw_df['lyrics_found'].sum()} / {len(raw_df)}")

    with_lyrics_path = os.path.join(DATA_DIR, "music_genre_dataset_with_lyrics.csv")
    raw_df.to_csv(with_lyrics_path, index=False)
    print(f"Saved to {with_lyrics_path}")

    # ---- Language filter ----
    print("\n=== Detecting language ===")
    raw_df["language"] = raw_df["lyrics"].apply(detect_language)
    print(raw_df["language"].value_counts(dropna=False))

    clean_df = raw_df[(raw_df["lyrics_found"]) & (raw_df["language"] == "en")].copy()
    print(f"\nEnglish + lyrics-found songs: {len(clean_df)}")
    print(clean_df["target_genre"].value_counts())

    # ---- Trim / backfill to hit ARTISTS_PER_GENRE_TARGET x SONGS_PER_ARTIST per genre ----
    print("\n=== Trimming to target artists/songs per genre ===")
    final_parts = []
    for genre in GENRE_TAGS:
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
