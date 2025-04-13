import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ml  
import numpy as np
import pandas as pd

# ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="WanderingMelody")

#spotify_df = pd.read_json("spotify-tracks-dataset.json")
lyric_df = pd.read_json("datasets/shortened_spotify.json")

# Preprocess the lyric data and build necessary indices
dict_of_lyrics = ml.lyric_df[['text']].to_dict(orient="index")
cleaned_tokenized_lyrics = ml.tokenize_lyrics(dict_of_lyrics, ml.tokenize)
cleaned_tokenized_lyrics = ml.remove_stop_words(ml.custom_stopwords, cleaned_tokenized_lyrics)
inverted_index = ml.build_inverted_index(cleaned_tokenized_lyrics)
clean_song_count = ml.compute_idf(inverted_index, len(cleaned_tokenized_lyrics))

@app.route("/recommendations")
def recommendations():
    print("Getting user inputs")
    mood = request.args.get("mood")
    location = request.args.get("location")
    age = request.args.get("age", default=18, type=int)
    weather = request.args.get("weather")  

    # Validate inputs (at least mood or genre should be provided)
    if not mood:
        return jsonify({"error": "Please provide a mood description"}), 400

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

    recommended_songs_with_scores, synonyms = ml.svd_recommend_songs(mood, cleaned_tokenized_lyrics, clean_song_count, age, inferred_weather_mood)

    print("finished svd rec songs function")
    if not recommended_songs_with_scores:
        return jsonify([])  
    
    song_details = ml.get_song_details_with_ratings(recommended_songs_with_scores)
    return jsonify(song_details)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)