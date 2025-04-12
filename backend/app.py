import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ml  
import numpy as np
import pandas as pd
# Import the scrape_google_places function
from scrape_google_places import get_google_maps_recommendations
import gc

# ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="WanderingMelody")

#spotify_df = pd.read_json("spotify-tracks-dataset.json")
#lyric_df = pd.read_json("spotify_millsongdata.json")

# Preprocess the lyric data and build necessary indices
dict_of_lyrics = ml.lyric_df[['text']].to_dict(orient="index")
cleaned_tokenized_lyrics = ml.tokenize_lyrics(dict_of_lyrics, ml.tokenize)
cleaned_tokenized_lyrics = ml.remove_stop_words(ml.custom_stopwords, cleaned_tokenized_lyrics)
inverted_index = ml.build_inverted_index(cleaned_tokenized_lyrics)
clean_song_count = ml.compute_idf(inverted_index, len(cleaned_tokenized_lyrics))

# Remove the refences to the memory-consuming datastructures after
# after they are done being used
del dict_of_lyrics, cleaned_tokenized_lyrics, inverted_index
# Manually call the garbage collector to clean up memory (just in case)
gc.collect()

@app.route("/recommendations")
def recommendations():
    print("Getting user inputs")
    mood = request.args.get("mood")
    location = request.args.get("location")
    age = request.args.get("age", default=18, type=int)
    genre = request.args.get("genre")
    weather = request.args.get("weather")  

    # Validate inputs (at least mood or genre should be provided)
    if not mood:
        return jsonify({"error": "Please provide at least a mood description or a genre."}), 400

    # map weather to mood descriptor
    weather_mood_map = {
        "sunny": "happy",
        "cloudy": "melancholy",
        "rainy": "sad",
        "stormy": "angry",
        "snowy": "calm",
        "foggy": "mysterious",
        "windy": "restless"
    }
    inferred_weather_mood = weather_mood_map.get(weather.lower(), "") if weather else ""

    print("calling svd rec songs function")
    # Call the recommendation function
    recommended_songs_with_scores, synonyms = ml.svd_recommend_songs(mood, cleaned_tokenized_lyrics, clean_song_count, age)

    print("finished svd rec songs function")
    if not recommended_songs_with_scores:
        return jsonify([])  

    song_details = ml.get_song_details_with_ratings(recommended_songs_with_scores)
    
    # Return song details as before to maintain compatibility
    return jsonify(song_details)

# separate endpoint for places
@app.route("/get_places")
def get_places():
    """Endpoint to get place recommendations based on location"""
    location = request.args.get("location")
    
    if not location or location.strip() == "":
        # Return empty array instead of error if no location
        return jsonify([])
    
    try:
        print(f"Getting place recommendations for {location}")
        places = get_google_maps_recommendations(location)
        print(f"Found {len(places)} places")
        return jsonify(places)
    except Exception as e:
        print(f"Error getting place recommendations: {e}")
        # Return empty array instead of error
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
