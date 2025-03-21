import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ml  

# ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))


app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def home():
    return render_template('base.html', title="WanderingMelody")

@app.route("/recommendations", methods=['GET'])
def recommendations():
    mood = request.args.get("mood")
    location = request.args.get("location")
    age = request.args.get("age", default=18, type=int)  # default age is 18 if not provided
    genre = request.args.get("genre")

    # Validate inputs (at least mood or genre should be provided)
    if not mood and not genre:
        return jsonify({"error": "Please provide at least a mood description or a genre."}), 400

    # Preprocess the lyric data and build necessary indices (using ML functions)
    dict_of_lyrics = ml.lyric_df[['text']].to_dict(orient="index")
    cleaned_tokenized_lyrics = ml.tokenize_lyrics(dict_of_lyrics, ml.tokenize)
    cleaned_tokenized_lyrics = ml.remove_stop_words(ml.custom_stopwords, cleaned_tokenized_lyrics)
    inverted_index = ml.build_inverted_index(cleaned_tokenized_lyrics)
    clean_song_count = ml.compute_idf(inverted_index, len(cleaned_tokenized_lyrics))

    # Call the recommendation function from ml.py
    recommended_songs, synonyms = ml.recommend_songs(genre, cleaned_tokenized_lyrics, clean_song_count)

    if not recommended_songs:
        return jsonify([])  
    
    # Get song details (e.g., title, artist, album, genre, rating)
    song_details = ml.get_song_details(recommended_songs)

    return jsonify(song_details)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
