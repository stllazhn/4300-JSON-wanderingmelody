import streamlit as st
import requests

def fetch_locations(query):
    """Fetches location suggestions using the Nominatim API and formats the response."""
    if not query:
        return []
    api_url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}"
    headers = {"User-Agent": "WanderingMelodyApp/1.0 (danishsafdaryan@gmail..com)"}  
    try:
        response = requests.get(api_url, headers=headers)
        print(f"Fetching locations for query: {query}")
        print(f"API URL: {api_url}")
        print(f"Response Status Code: {response.status_code}")
        if response.status_code == 200:
            locations = response.json()
            print(f"Raw API Response: {locations}")
            formatted_locations = [f"{loc['display_name']}" for loc in locations]
            print(f"Formatted Locations: {formatted_locations}")
            return formatted_locations
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching locations: {e}")
    return []

def search_recommendations(mood, location, age, genre):
    """Simulates generating music recommendations based on input criteria."""
    if not location or not age or not genre:
        return "Please fill in all fields."
    return f"🎵 Here are your personalized music recommendations for {mood} in {location}, age {age}, genre {genre}."

# Streamlit UI
st.title("🎶 WanderingMelody")
st.write("Describe your mood or trip experience and get music recommendations!")

# Input fields
mood = st.text_area("Describe your mood or trip experience")
location_query = st.text_input("Search Location")
location_suggestions = fetch_locations(location_query)
selected_location = st.selectbox("Select a Location", location_suggestions) if location_suggestions else location_query
age = st.number_input("Enter Age", min_value=1, max_value=100, step=1)
genre = st.text_input("Enter Music Genre")

if st.button("Search"):
    result = search_recommendations(mood, selected_location, age, genre)
    st.success(result)
