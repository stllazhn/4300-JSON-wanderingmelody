import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load the data
lyric_df = pd.read_json("spotify_millsongdata.json")

# Define the function
def polarity_scores_for_songs(df):
    sia = SentimentIntensityAnalyzer() 
    df['sentiment'] = df['text'].apply(lambda song_lyrics: sia.polarity_scores(song_lyrics)['compound'])
    return df

# Apply the function
lyric_df = polarity_scores_for_songs(lyric_df)

# Save the updated DataFrame back to JSON
lyric_df.to_json("spotify_millsongdata.json", orient='records', indent=2)