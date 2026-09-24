"""
Scale-check script: probes how many eligible artists/songs each genre
yields under the UPDATED thresholds (>=1000 ListenBrainz listeners,
wider MusicBrainz candidate pool) before committing to the full
400-song (100/genre, 20 artists x 5 songs) collection run.

Not the final pipeline -- just answers "is 400 achievable?" and reports
where each genre's attrition happens (MusicBrainz eligibility ->
popularity cutoff -> lyrics found -> English-only).
"""

import requests
import pandas as pd
import time
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "DS4002GenreProject/1.0 (kfg2ec@virginia.edu)"}
MB_BASE_URL = "https://musicbrainz.org/ws/2/recording"
LB_URL = "https://api.listenbrainz.org/1/popularity/artist"
LRCLIB_URL = "https://lrclib.net/api/search"

MIN_LISTENERS = 1000
SONGS_PER_ARTIST_NEEDED = 5
ARTISTS_PER_GENRE_TARGET = 20
CANDIDATE_PAGES = 5  # 5 x 100 = 500 raw recordings per genre tag (MusicBrainz caps offset+limit at 500)

GENRE_TAGS = {
    "Country": ["country"],
    "Pop": ["pop"],
    "Rock": ["rock"],
    "Hip-Hop/Rap": ["hip hop"],
}

# Broadened variants for the two genres that fell short of 20 artists
# with a single tag search (Pop: 17, Hip-Hop/Rap: 12).
GENRE_TAGS_WIDE = {
    "Pop": ["pop", "pop rock", "dance-pop", "electropop", "synthpop"],
    "Hip-Hop/Rap": ["hip hop", "rap", "hip-hop", "gangsta rap", "trap"],
}


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
            print(f"    MusicBrainz busy, waiting 5s (attempt {attempt+1}/{max_attempts})")
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
            break  # MusicBrainz caps offset+limit at 500
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


def probe_genre(search_tags, target_genre):
    if isinstance(search_tags, str):
        search_tags = [search_tags]

    print(f"\n=== {target_genre} (tags: {search_tags}) ===")

    all_recordings = []
    for tag in search_tags:
        print(f"  -- querying tag '{tag}' --")
        recordings = fetch_candidates(tag)
        print(f"     raw recordings fetched: {len(recordings)}")
        all_recordings.extend(recordings)
        time.sleep(1.5)

    df = recordings_to_df(all_recordings, target_genre)
    print(f"  Unique recordings across all tags: {len(df)}")
    if df.empty:
        return None

    # Song count per artist (>= songs needed, so there's slack for lyric misses)
    artist_counts = df["artist"].value_counts()
    eligible_song_count = artist_counts[artist_counts >= SONGS_PER_ARTIST_NEEDED]
    print(f"  Artists with >= {SONGS_PER_ARTIST_NEEDED} songs: {len(eligible_song_count)}")

    # Single-artist credit only (no collaborations)
    artist_df = (
        df[df["artist"].isin(eligible_song_count.index)][["artist", "artist_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    artist_df = artist_df[~artist_df["artist_id"].str.contains(",", na=False)].copy()
    artist_df = artist_df[artist_df["artist_id"] != ""].copy()
    print(f"  ...of which single-artist-credit: {len(artist_df)}")

    if artist_df.empty:
        return {"genre": target_genre, "df": df, "eligible_artists": artist_df}

    # ListenBrainz popularity
    listen_counts, user_counts = [], []
    for i, artist_id in enumerate(artist_df["artist_id"]):
        listens, users = get_listenbrainz_popularity(artist_id)
        listen_counts.append(listens)
        user_counts.append(users)
        time.sleep(0.5)
        if (i + 1) % 20 == 0:
            print(f"    checked popularity for {i+1}/{len(artist_df)} artists...")

    artist_df["listen_count"] = listen_counts
    artist_df["user_count"] = user_counts

    popular = artist_df[artist_df["user_count"].fillna(0) >= MIN_LISTENERS].copy()
    print(f"  Artists with >= {MIN_LISTENERS:,} ListenBrainz listeners: {len(popular)}")
    print(f"  --> Need {ARTISTS_PER_GENRE_TARGET} artists x {SONGS_PER_ARTIST_NEEDED} songs "
          f"= {ARTISTS_PER_GENRE_TARGET * SONGS_PER_ARTIST_NEEDED} songs for this genre")

    status = "OK" if len(popular) >= ARTISTS_PER_GENRE_TARGET else "SHORT"
    print(f"  STATUS: {status} ({len(popular)}/{ARTISTS_PER_GENRE_TARGET} artists available)")

    return {
        "genre": target_genre,
        "df": df,
        "artist_pool": artist_df,
        "popular_artists": popular,
    }


def main():
    # Re-probe only Pop and Hip-Hop/Rap with broadened tag variants --
    # Country (22 artists) and Rock (24 artists) already cleared the
    # 20-artist bar with a single tag in the previous run.
    genres_to_run = GENRE_TAGS_WIDE

    results = {}
    for target_genre, tags in genres_to_run.items():
        try:
            results[target_genre] = probe_genre(tags, target_genre)
        except Exception as e:
            print(f"  ERROR probing {target_genre}: {e}")
            results[target_genre] = None
        time.sleep(2)

    print("\n\n=== WIDE-TAG SUMMARY (Pop, Hip-Hop/Rap only) ===")
    summary_rows = []
    for genre, res in results.items():
        if res is None or res.get("popular_artists") is None:
            summary_rows.append({"genre": genre, "eligible_artists": 0, "max_songs_at_5_each": 0})
            continue
        n_artists = len(res["popular_artists"])
        summary_rows.append({
            "genre": genre,
            "eligible_artists": n_artists,
            "max_songs_at_5_each": n_artists * SONGS_PER_ARTIST_NEEDED,
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    out_path = os.path.join(DATA_DIR, "scale_check_summary_wide.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved wide-tag summary to {out_path}")

    # Combine with the single-tag Country/Rock results from the first run
    prior_path = os.path.join(DATA_DIR, "scale_check_summary.csv")
    if os.path.exists(prior_path):
        prior_df = pd.read_csv(prior_path)
        prior_df = prior_df[prior_df["genre"].isin(["Country", "Rock"])]
        combined_df = pd.concat([prior_df, summary_df], ignore_index=True)
        combined_df = combined_df.set_index("genre").loc[["Country", "Pop", "Rock", "Hip-Hop/Rap"]].reset_index()
        combined_path = os.path.join(DATA_DIR, "scale_check_summary_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print("\n=== COMBINED SUMMARY (all 4 genres) ===")
        print(combined_df.to_string(index=False))
        print(f"\nSaved combined summary to {combined_path}")

    # Save the full popular-artist pools too, for inspection
    for genre, res in results.items():
        if res is None or res.get("popular_artists") is None:
            continue
        safe_name = genre.replace("/", "-").replace(" ", "_").lower()
        pool_path = os.path.join(DATA_DIR, f"scale_check_{safe_name}_artist_pool_wide.csv")
        res["popular_artists"].sort_values("user_count", ascending=False).to_csv(pool_path, index=False)
        print(f"Saved {genre} artist pool to {pool_path}")


if __name__ == "__main__":
    main()
