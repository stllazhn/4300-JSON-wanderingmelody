import streamlit as st
import requests
import ml

st.title("🎶 WanderingMelody")
st.write("Describe your mood or trip experience and get music recommendations!")

# Input fields
mood = st.text_area("Describe your mood or trip experience")
location_query = st.text_input("Search Location (Optional)")
age = st.number_input("Enter Age (Optional)", min_value=1, max_value=100, step=1, value=18)
genre = st.text_input("Enter Music Genre")

if st.button("Search"):
    if not mood and not genre:
        st.error("Please provide at least a mood description or a genre.")
    else:
        # Call ML function
        recommended_songs, synonyms = ml.recommend_songs(genre, {}, {})  # Using empty dicts for now
        if recommended_songs:
            st.success(f"🎵 Recommended Songs: {', '.join(map(str, recommended_songs))}")
        else:
            st.warning("No songs found, try another genre or mood.")
