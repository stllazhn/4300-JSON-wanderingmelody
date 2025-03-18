import time
import lyricsgenius
from datasets import load_dataset
import pandas as pd

# Your Genius API client access token
client_access_token = 'TM4Zzop0aFW8mzPthc9r6qX0Y3ICwhXkjhEl7YObla043NT3qGI1eMwyIphSjVP_'

# Initialize the Genius client with a higher timeout
genius = lyricsgenius.Genius(
  client_access_token,
  remove_section_headers=True,  # Removes section headers like [Chorus], [Verse], etc.
  skip_non_songs=True,          # Skips non-song results (e.g., albums, artists)
  verbose=False,                # Set to True for debugging output
  timeout=15                    # Increase timeout to 15 seconds
)

# Function to search for song lyrics
def get_song_lyrics(song_name, artist_name, max_retries=3):
  retries = 0
  while retries < max_retries:
    try:
      # Search for the song
      song = genius.search_song(song_name, artist_name)
      if song:
        return song.lyrics
      else:
        return "Song not found."
    except Exception as e:
      retries += 1
      if "timed out" in str(e):  # Handle timeout errors
        print(f"Request timed out. Retrying ({retries}/{max_retries})...")
        time.sleep(5)  # Wait for 5 seconds before retrying
      elif "Too Many Requests" in str(e):  # Handle rate limiting
        print("Rate limit exceeded. Retrying after 10 seconds...")
        time.sleep(10)  # Wait for 10 seconds before retrying
      else:
        return f"An error occurred: {e}"
  return "Max retries reached. Please try again later."

# Function to load the Hugging Face dataset and the CSV file
def load_and_merge_datasets(dataset_path, csv_path):
  # Load Hugging Face dataset
  dataset = load_dataset(dataset_path)
  df1 = pd.DataFrame(dataset['train'])
  
  # Load the CSV file containing the second dataset
  df2 = pd.read_csv(csv_path)
  
  # Combine song names and artists from both datasets for comparison
  df1_combined = df1['track_name'] + " - " + df1['artists']
  df2_combined = df2['song'] + " - " + df2['artist']
  
  # Find missing songs
  missing_songs = df1[~df1_combined.isin(df2_combined)]
  return missing_songs

# Function to display missing songs and lyrics
def process_missing_songs(missing_songs, max_lyrics_retries=3):
  for index, row in missing_songs.iterrows():
    song_name = row['track_name']
    artist_name = row['artists']
    
    print(f"Fetching lyrics for {song_name} by {artist_name}...")
    lyrics = get_song_lyrics(song_name, artist_name, max_retries=max_lyrics_retries)
    print(f"Lyrics for {song_name}:\n{lyrics}\n{'-'*50}")

# Main function
def main():
  # Example paths for the Hugging Face dataset and CSV file
  dataset_path = "maharshipandya/spotify-tracks-dataset"
  csv_path = "spotify_millsongdata.csv"  # Replace with the actual path to your CSV file
  
  # Load and merge datasets
  missing_songs = load_and_merge_datasets(dataset_path, csv_path)
  
  # Display missing songs and fetch their lyrics
  process_missing_songs(missing_songs)

# Run the main function
if __name__ == "__main__":
  main()
