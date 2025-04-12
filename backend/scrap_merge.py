import os
import time
import json
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import lyricsgenius
import threading
import re
from langdetect import detect, LangDetectException
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import requests
from typing import Dict, Any, Optional, List
import urllib.parse

# Genius API access token
GENIUS_ACCESS_TOKEN = 'h2ASjYK2uM4gQdafPc5_ONB4yn4lSJKI8jBEPH8s47oa1_N8DkLwHbsAoax6bzmP'
genius = lyricsgenius.Genius(
    GENIUS_ACCESS_TOKEN,
    remove_section_headers=True,
    skip_non_songs=True,
    verbose=False,
    timeout=30
)

# Spotify API setup
SPOTIFY_CLIENT_ID = 'ddc05c0eb8d049b0878d979f2edb6812'
SPOTIFY_CLIENT_SECRET = '25cc51dee16341a0b510fecf4e23911a'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    requests_timeout=30,
    requests_session=True
))

# Cache setup
LYRICS_CACHE_FILE = "lyrics_cache.json"
SPOTIFY_CACHE_FILE = "spotify_cache.json"
OUTPUT_FILE = "combined_songs_with_lyrics.csv"
cache_lock = threading.Lock()

def load_cache(cache_file: str) -> Dict[str, Any]:
    """Load cache from file if exists"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache file {cache_file}: {e}")
    return {}

def save_cache(cache: Dict[str, Any], cache_file: str) -> None:
    """Save cache to file"""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except IOError as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")

# Initialize caches
lyrics_cache = load_cache(LYRICS_CACHE_FILE)
spotify_cache = load_cache(SPOTIFY_CACHE_FILE)

def verify_spotify_connection() -> bool:
    """Verify Spotify API connection"""
    try:
        sp.track('6rqhFgbbKwnb9MLmUQDhG6')  # Test with a known track ID
        print("✅ Spotify connection verified")
        return True
    except Exception as e:
        print(f"❌ Spotify connection failed: {e}")
        return False
    
def load_and_merge_datasets(dataset_path: str, csv_path: str) -> pd.DataFrame:
    """Load and merge datasets keeping only artist & track name"""
    try:
        # Load Spotify dataset
        spotify_df = pd.DataFrame(load_dataset(dataset_path)['train'])
        spotify_df = spotify_df[['artists', 'track_name']].rename(columns={'artists': 'artist'})
        
        # Load lyrics CSV
        lyrics_df = pd.read_csv(csv_path)
        lyrics_df = lyrics_df[['artist', 'song', 'text']].rename(columns={'song': 'track_name', 'text': 'lyrics'})
        
        # Combine and deduplicate
        combined_df = pd.concat([spotify_df, lyrics_df[['artist', 'track_name']]], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['artist', 'track_name'])
        
        return combined_df, lyrics_df
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

def clean_lyrics_text(lyrics: str) -> str:
    """Clean lyrics text by removing metadata patterns and newlines"""
    if not isinstance(lyrics, str):
        return ""
        
    patterns = [
        r'\d+\s*contributors?',
        r'lyrics',
        r'song\s*discussion',
        r'translation',
        r'see\s*also',
        r'embed',
        r'Read More'
    ]
    
    split_pattern = re.compile('|'.join(patterns), flags=re.IGNORECASE)
    matches = list(split_pattern.finditer(lyrics))
    
    if matches:
        last_match = matches[-1]
        end_pos = last_match.end()
        while end_pos < len(lyrics) and (lyrics[end_pos].isspace() or lyrics[end_pos] in [':', '-', '–', '.', ',']):
            end_pos += 1
        cleaned = lyrics[end_pos:]
    else:
        cleaned = lyrics
    
    # Remove newlines and clean up whitespace
    cleaned = re.sub(r'\n+', ' ', cleaned)  # Replace newlines with spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
    return cleaned.strip()

def get_song_lyrics(song_name: str, artist_name: str, existing_lyrics: pd.DataFrame, retries: int = 3) -> str:
    """Get lyrics from Genius with caching and retries, or from existing CSV if available"""
    key = f"{song_name} - {artist_name}".lower()

    # First check if we have lyrics in the existing CSV
    existing = existing_lyrics[
        (existing_lyrics['track_name'].str.lower() == song_name.lower()) & 
        (existing_lyrics['artist'].str.lower() == artist_name.lower())
    ]
    
    if not existing.empty:
        lyrics = existing.iloc[0]['lyrics']
        if pd.notna(lyrics) and lyrics.strip() != "":
            with cache_lock:
                lyrics_cache[key] = lyrics
                save_cache(lyrics_cache, LYRICS_CACHE_FILE)
            return lyrics

    # Then check cache
    with cache_lock:
        if key in lyrics_cache:
            return lyrics_cache[key]

    # Finally, try Genius API
    for attempt in range(retries):
        try:
            song = genius.search_song(song_name, artist_name, False)
            if song is None:
                lyrics = "Song not found."
            else:
                lyrics = song.lyrics if hasattr(song, 'lyrics') else "Lyrics not available."
                lyrics = clean_lyrics_text(lyrics)

            with cache_lock:
                lyrics_cache[key] = lyrics
                save_cache(lyrics_cache, LYRICS_CACHE_FILE)
            return lyrics

        except Exception as e:
            wait_time = 10 if "Too Many Requests" in str(e) else 5
            time.sleep(wait_time)
            if attempt == retries - 1:
                error_msg = f"Error: {e}" if "Error:" not in str(e) else str(e)
                with cache_lock:
                    lyrics_cache[key] = error_msg
                    save_cache(lyrics_cache, LYRICS_CACHE_FILE)
                return error_msg

def fetch_lyrics_row(row: pd.Series, existing_lyrics: pd.DataFrame) -> Dict[str, Any]:
    """Fetch lyrics for a single row"""
    lyrics = get_song_lyrics(row['track_name'], row['artist'], existing_lyrics)
    return {
        'song': row['track_name'],
        'artist': row['artist'],
        'lyrics': lyrics
    }

def process_lyrics(missing_songs: pd.DataFrame, existing_lyrics: pd.DataFrame, max_workers: int = 1000) -> pd.DataFrame:
    """Process lyrics for multiple songs using ThreadPoolExecutor"""
    lyrics_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_lyrics_row, row, existing_lyrics): idx for idx, row in missing_songs.iterrows()}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                lyrics_data.append(result)
            except Exception as e:
                print(f"[{i}] Error in future: {e}")
            
            if i % 10 == 0:
                print(f"Fetched {i}/{len(futures)} lyrics...")
    
    return pd.DataFrame(lyrics_data)

def get_spotify_track_data(artist_name: str, track_name: str) -> Optional[Dict[str, Any]]:
    """Get Spotify track data with error handling"""
    try:
        results = sp.search(
            q=f'artist:{artist_name} track:{track_name}',
            type='track',
            limit=1
        )
        print(results)
        if results['tracks']['items']:
            print("hi")
            track = results['tracks']['items'][0]
            return {
                'spotify_id': track["id"],
                'available_markets': track["available_markets"],
                'image_url': track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                'popularity': track["popularity"],
                'song_url': track["external_urls"]["spotify"]
            }
    except Exception as e:
        print(f"Error searching {artist_name} - {track_name}: {e}")
    return None

def process_spotify_data(lyrics_df: pd.DataFrame) -> pd.DataFrame:
    """Process Spotify data for songs with lyrics, with rate limiting"""
    spotify_data = []
    api_calls_made = 0
    max_api_calls_before_sleep = 60  # Sleep after this many API calls
    sleep_duration = 60  # Sleep for 60 seconds
    
    # Create progress bar
    pbar = tqdm(total=len(lyrics_df), desc="Processing Spotify data")
    
    for idx, row in lyrics_df.iterrows():
        # Check cache first
        cache_key = f"{row['artist']} - {row['song']}".lower()
        with cache_lock:
            if cache_key in spotify_cache:
                spotify_data.append(spotify_cache[cache_key])
                pbar.update(1)
                continue

        # Make API call if not in cache
        track_data = get_spotify_track_data(artist_name=row['artist'], track_name=row['song'])
        api_calls_made += 1
        
        if track_data:
            data = {
                'song': row['song'],
                'artist': row['artist'],
                'lyrics': row['lyrics'],
                **track_data
            }
            
            with cache_lock:
                spotify_cache[cache_key] = data
                save_cache(spotify_cache, SPOTIFY_CACHE_FILE)
            
            spotify_data.append(data)
        
        pbar.update(1)
        
        # Rate limiting
        if api_calls_made % max_api_calls_before_sleep == 0:
            print(f"\nMade {api_calls_made} API calls. Sleeping for {sleep_duration} seconds...")
            time.sleep(sleep_duration)
        
        # Small delay between calls
        time.sleep(0.1)
    
    pbar.close()
    print(f"Total API calls made: {api_calls_made}")
    return pd.DataFrame(spotify_data)

def is_english(text: str) -> bool:
    """Check if text is in English"""
    if not isinstance(text, str) or len(text) < 20:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def clean_combined_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the combined dataframe"""
    if df.empty:
        return df
    
    # Remove error messages
    error_patterns = ["Max retries reached", "Error:", "Song not found", "Lyrics not available"]
    clean_df = df[~df['lyrics'].str.contains('|'.join(error_patterns), case=False, na=False)]
    
    # Filter English lyrics and clean text
    clean_df = clean_df[clean_df['lyrics'].apply(is_english)]
    clean_df['lyrics'] = clean_df['lyrics'].apply(clean_lyrics_text)
    
    return clean_df.drop_duplicates(subset=['artist', 'song'])

def clean_lyrics_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean lyrics dataframe by removing errors and empty entries"""
    if df.empty:
        return df
    
    # Remove error messages and empty lyrics
    error_patterns = [
        "Max retries reached", 
        "Error:", 
        "Song not found", 
        "Lyrics not available",
        "[Errno 429] 429 Client Error"
    ]
    
    # Filter out rows with error patterns or empty lyrics
    mask = (
        ~df['lyrics'].str.contains('|'.join(error_patterns), case=False, na=False) &
        df['lyrics'].notna() &
        (df['lyrics'].str.strip() != "")
    )
    
    # 1. Filter rows IN-PLACE using boolean indexing
    df.drop(index=df[~mask].index, inplace=True)
    
    # 2. Modify lyrics column IN-PLACE
    df['lyrics'] = df['lyrics'].apply(clean_lyrics_text)  # Now safe because we modified the original
    
    # 3. Filter English lyrics IN-PLACE
    english_mask = df['lyrics'].apply(is_english)
    df.drop(index=df[~english_mask].index, inplace=True)
    
    return df

def main():
    if not verify_spotify_connection():
        return

    dataset_path = "maharshipandya/spotify-tracks-dataset"
    lyrics_path = "spotify_millsongdata.csv"

    print("Loading and merging datasets...")
    try:
        merged_df, existing_lyrics_df = load_and_merge_datasets(dataset_path, lyrics_path)
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return
    
    print(f"\nFound {len(merged_df)} unique songs")

    print("Fetching missing lyrics...")
    # Process and clean lyrics first
    lyrics_df = process_lyrics(merged_df, existing_lyrics_df)
    cleaned_lyrics_df = clean_lyrics_data(lyrics_df)

    print("\nFetching Spotify data for songs with lyrics...")
    spotify_df = process_spotify_data(cleaned_lyrics_df)

    print("\nCombining and cleaning data...")
    cleaned_df = clean_combined_data(spotify_df)

    print(f"\nSaving results to {OUTPUT_FILE}")
    try:
        cleaned_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Final dataset contains {len(cleaned_df)} songs")
    except Exception as e:
        print(f"Failed to save results: {e}")

def cleanJson():
    df = pd.read_csv("lyrics.csv")

    df = process_spotify_data(df)
    print(f"\nSaving results to {OUTPUT_FILE}")
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Final dataset contains {len(df)} songs")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    # main()
    cleanJson()
