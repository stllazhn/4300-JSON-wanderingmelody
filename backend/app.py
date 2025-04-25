import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ml  
import numpy as np
import pandas as pd
from helpers.location_review_similarity import find_similar_locations, load_reviews_database

# ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

app = Flask(__name__)
CORS(app)

# Load the reviews database once at startup
REVIEWS_DATABASE = load_reviews_database()

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

    recommended_songs_with_scores, synonyms, topic_matches, ideal_sentiment, common_word_scores, antonym_word_scores, weather_word_scores, popularity_word_scores = ml.svd_recommend_songs(mood, cleaned_tokenized_lyrics, clean_song_count, age, inferred_weather_mood)

    print("finished svd rec songs function")
    if not recommended_songs_with_scores:
        return jsonify([])  
    
    song_details = ml.get_song_details_with_ratings(recommended_songs_with_scores,topic_matches, common_word_scores, antonym_word_scores, weather_word_scores)
    for song in song_details:
        for key, value in song.items():
            if isinstance(value, (np.integer, np.floating)):
                song[key] = value.item()  # Convert numpy type to native Python type
                
    return jsonify(song_details)

@app.route("/similar_locations")
def similar_locations():
    """
    API endpoint to get similar locations based on mood and country
    """
    country = request.args.get("country", "")
    mood = request.args.get("mood", "")
    
    print(f"Received request for similar locations - Country: {country}, Mood: {mood}")
    
    if not country or not mood:
        return jsonify([])
    
    # Get the top 3 similar locations
    try:
        similar_locs = find_similar_locations(country, mood)
        print(f"Found similar locations: {similar_locs}")
        
        # Find the country data in the reviews database
        country_data = None
        for data in REVIEWS_DATABASE:
            if data["country"].lower() == country.lower():
                country_data = data
                break
        
        if not country_data:
            return jsonify({"error": f"Country data not found for {country}"}), 404
        
        # Format the response with full location data
        locations = []
        for i, (location_name, similarity) in enumerate(similar_locs):
            if location_name != "No locations found" and location_name != "No location found":
                # Find the index of this location in the country data
                location_index = None
                for j in range(1, 6):
                    if country_data[f"location_{j}"] == location_name:
                        location_index = j
                        break
                
                if location_index:
                    locations.append({
                        "name": location_name,
                        "rating": country_data[f"rating_{location_index}"],
                        "num_reviews": country_data[f"num_reviews_{location_index}"],
                        "top_review": country_data[f"top_review_{location_index}"],
                        "similarity": similarity
                    })
        
        return jsonify(locations)
    except Exception as e:
        print(f"Error finding similar locations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
