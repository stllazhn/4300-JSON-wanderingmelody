import streamlit as st
import ml

st.title("🎶 WanderingMelody")
st.write("Describe your mood or trip experience and get music recommendations!")

# Input fields
mood = st.text_area("Describe your mood or trip experience")
location_query = st.text_input("Search Location (Optional)")
age = st.number_input("Enter Age (Optional)", min_value=1, max_value=100, step=1, value=18)
genre = st.text_input("Enter Music Genre")

# Preprocess Lyrics Dataset (Move this from ml.py to app.py)
dict_of_lyrics = ml.lyric_df[['text']].to_dict(orient="index")

# Tokenize lyrics
cleaned_tokenized_lyrics = ml.tokenize_lyrics(dict_of_lyrics, ml.tokenize)

# Remove stopwords
cleaned_tokenized_lyrics = ml.remove_stop_words(ml.custom_stopwords, cleaned_tokenized_lyrics)

# Build inverted index for faster searching
inverted_index = ml.build_inverted_index(cleaned_tokenized_lyrics)

# Compute clean song count (IDF calculation)
clean_song_count = ml.compute_idf(inverted_index, len(cleaned_tokenized_lyrics))

if st.button("Search"):
    if not mood and not genre:
        st.error("Please provide at least a mood description or a genre.")
    else:
        # Call ML function
        recommended_songs, synonyms = ml.recommend_songs(genre, cleaned_tokenized_lyrics, clean_song_count)
        
        if recommended_songs:
            # Get actual song details instead of raw IDs
            song_details = ml.get_song_details(recommended_songs)

            if song_details:
                st.write("### Recommended Songs:")
                for song in song_details:
                    st.write(f"🎵 **{song['title']}** by *{song['artist']}*")
                    st.write(f"   🎼 Album: {song['album']} | 🎤 Genre: {song['genre']}")
                    st.write("---")  # Adds a separator for readability
        else:
            st.warning("No songs found, try another genre or mood.")
