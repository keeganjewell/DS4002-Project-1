"""
PIPELINE STEP 4 of 6: full dataset collection (initial data -> raw data).

Full dataset collection pipeline for the lyrics/genre classification project.

For each of Country, Pop, Rock, Hip-Hop/Rap:
  1. Query MusicBrainz (broadened tag variants for Pop/Rock/Hip-Hop/Rap,
     single tag for Country -- see 03_scale_check.py for why) for candidate
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

NOTE: on its own, this script's single "country" tag search under-fills
Country (only ~13-15 eligible artists instead of 20). This is expected --
run 05_backfill_country.py immediately afterward to widen the Country tag
search and bring it up to the full 20 artists / 100 songs. Steps 04 and 05
together are what produce the final, class-balanced dataset.

Run with: python SCRIPTS/04_collect_dataset.py
Requires: internet access (MusicBrainz, ListenBrainz, LRCLIB APIs); takes
          on the order of tens of minutes due to API rate limiting.
Output: DATA/music_genre_dataset_raw.csv (all selected songs, pre-lyrics)
        DATA/music_genre_dataset_with_lyrics.csv (post-LRCLIB, pre-language-filter)
        DATA/music_genre_dataset_final.csv (English-only, lyrics-found -- the modeling dataset,
                                             still Country-short until step 05 runs)
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
    "Country": ["country"],
    "Pop": ["pop", "pop rock", "dance-pop", "electropop", "synthpop"],
    "Rock": ["rock", "classic rock", "alternative rock", "hard rock", "indie rock"],
    "Hip-Hop/Rap": ["hip hop", "rap", "hip-hop", "gangsta rap", "trap"],
}


# --------------------------------------------------------------------------
# MusicBrainz
# --------------------------------------------------------------------------

def get_musicbrainz_data(params, max_attempts=5):
    """GET a single page of MusicBrainz recording search results, retrying
    on transient errors (503 busy, network errors) up to max_attempts times."""
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
            # MusicBrainz's shared rate-limit pool is temporarily saturated; back off and retry.
            print(f"    MusicBrainz busy, waiting 5s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(5)
        else:
            print(f"    unexpected status {response.status_code}: {response.text[:200]}")
            break
    raise Exception("MusicBrainz request failed after multiple attempts.")


def fetch_candidates(search_tag, pages=CANDIDATE_PAGES):
    """Page through MusicBrainz recording search results for one tag,
    100 results per page, up to `pages` pages (MusicBrainz caps
    offset+limit at 500, so 5 pages of 100 is the maximum reachable)."""
    all_recordings = []
    for page in range(pages):
        offset = page * 100
        if offset + 100 > 500:
            break  # MusicBrainz refuses offset+limit > 500
        params = {"query": f'tag:"{search_tag}"', "fmt": "json", "limit": 100, "offset": offset}
        data = get_musicbrainz_data(params)
        recs = data.get("recordings", [])
        if not recs:
            break  # ran out of results before hitting the page cap
        all_recordings.extend(recs)
        time.sleep(1.1)  # stay under MusicBrainz's ~1 request/second rate limit
    return all_recordings


def recordings_to_df(recordings, target_genre):
    """Flatten raw MusicBrainz recording JSON into a tidy DataFrame, one
    row per recording, keeping only the fields the pipeline needs."""
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
            # Multiple artist-credits (features/collabs) get comma-joined here;
            # these get filtered out later since we only want single-artist songs.
            "artist": ", ".join(artist_names),
            "artist_id": ", ".join(artist_ids),
            "musicbrainz_tags": ", ".join(tags),
            "target_genre": target_genre,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # The same recording can surface from multiple tag searches (e.g. a
    # song tagged both "pop" and "dance-pop"); de-duplicate by MBID.
    return df.drop_duplicates(subset="recording_id").reset_index(drop=True)


# --------------------------------------------------------------------------
# ListenBrainz
# --------------------------------------------------------------------------

def get_listenbrainz_popularity(artist_mbid, max_attempts=3):
    """Look up an artist's total ListenBrainz listen count and unique
    listener count by MusicBrainz artist ID, used as the popularity filter
    (>= MIN_LISTENERS) so we don't pull extremely obscure artists."""
    payload = {"artist_mbids": [artist_mbid]}
    for attempt in range(max_attempts):
        try:
            response = requests.post(LB_URL, json=payload, timeout=30)
        except requests.RequestException:
            time.sleep(2)
            continue
        if response.status_code == 429:  # rate limited
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
    """Search LRCLIB for a song by title + artist and return its best
    match's (matched_artist, matched_title, plain-text lyrics). Returns
    (None, None, None) if no match is found or the request ultimately fails."""
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
            # Respect the server's requested backoff instead of a fixed sleep.
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

    # Take LRCLIB's top-ranked match; plainLyrics strips timing/LRC markup.
    result = results[0]
    return result.get("artistName"), result.get("trackName"), result.get("plainLyrics")


def detect_language(text):
    """Detect the primary language of a lyrics string, used to keep only
    English-language songs. Returns None for missing/empty text or if
    langdetect can't confidently classify it."""
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
    """For one genre, query MusicBrainz across all of its tag variants,
    then filter down to artists who (a) have enough songs in the pool,
    (b) are credited solo (no multi-artist collaborations), and (c) clear
    the ListenBrainz popularity threshold. Returns (all candidate
    recordings, eligible+popular artist table)."""
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
    # Drop multi-artist credits (comma-joined artist_id means a collab/feature)
    # and any row where MusicBrainz didn't resolve an artist MBID at all.
    artist_df = artist_df[~artist_df["artist_id"].str.contains(",", na=False)].copy()
    artist_df = artist_df[artist_df["artist_id"] != ""].copy()
    print(f"  Artists with >={SONGS_PER_ARTIST} songs, single-credit: {len(artist_df)}")

    # Look up each remaining artist's ListenBrainz popularity one at a time
    # (the API only accepts one artist per request).
    listen_counts, user_counts = [], []
    for i, artist_id in enumerate(artist_df["artist_id"]):
        listens, users = get_listenbrainz_popularity(artist_id)
        listen_counts.append(listens)
        user_counts.append(users)
        time.sleep(0.5)  # stay well under ListenBrainz's rate limit
        if (i + 1) % 25 == 0:
            print(f"    ListenBrainz checked {i + 1}/{len(artist_df)}")

    artist_df["listen_count"] = listen_counts
    artist_df["user_count"] = user_counts

    # Keep only artists with at least MIN_LISTENERS unique ListenBrainz
    # listeners; treat a missing/unknown user_count as 0 (fails the filter).
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
    # ---- Stage 1: build a candidate artist/song pool per genre ----
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

    # Combine all four genres' candidate songs into one raw pool and save
    # a checkpoint before the (slow, rate-limited) lyrics-fetching stage.
    raw_df = pd.concat(all_selections, ignore_index=True)
    raw_path = os.path.join(DATA_DIR, "music_genre_dataset_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw selection ({len(raw_df)} songs) to {raw_path}")
    print(raw_df["target_genre"].value_counts())

    # ---- Stage 2: fetch lyrics for every candidate song from LRCLIB ----
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
            # Periodic checkpoint so a crash/interruption during this long,
            # rate-limited loop doesn't lose all progress made so far.
            checkpoint = raw_df.iloc[:i + 1].copy()
            checkpoint["lrclib_artist"] = matched_artists
            checkpoint["lrclib_title"] = matched_titles
            checkpoint["lyrics"] = lyrics_list
            checkpoint.to_csv(os.path.join(DATA_DIR, "music_genre_dataset_checkpoint.csv"), index=False)
        time.sleep(1)  # stay under LRCLIB's rate limit

    raw_df["lrclib_artist"] = matched_artists
    raw_df["lrclib_title"] = matched_titles
    raw_df["lyrics"] = lyrics_list
    raw_df["lyrics_found"] = raw_df["lyrics"].notna()

    print(f"Lyrics found: {raw_df['lyrics_found'].sum()} / {len(raw_df)}")

    # One retry pass for misses -- a song can fail the first time due to a
    # transient network/rate-limit issue rather than genuinely not existing
    # on LRCLIB, so it's worth trying each miss exactly once more.
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

    # ---- Stage 3: keep only English-language songs with lyrics found ----
    print("\n=== Detecting language ===")
    raw_df["language"] = raw_df["lyrics"].apply(detect_language)
    print(raw_df["language"].value_counts(dropna=False))

    clean_df = raw_df[(raw_df["lyrics_found"]) & (raw_df["language"] == "en")].copy()
    print(f"\nEnglish + lyrics-found songs: {len(clean_df)}")
    print(clean_df["target_genre"].value_counts())

    # ---- Stage 4: trim each genre down to exactly ARTISTS_PER_GENRE_TARGET
    # artists x SONGS_PER_ARTIST songs, so every genre is equally represented
    # in the final modeling dataset ----
    print("\n=== Trimming to target artists/songs per genre ===")
    final_parts = []
    for genre in GENRE_TAGS:
        genre_clean = clean_df[clean_df["target_genre"] == genre]
        artist_song_counts = genre_clean.groupby("artist").size()
        # Only artists with a full SONGS_PER_ARTIST surviving English songs
        # are usable, so every chosen artist contributes the same count.
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
