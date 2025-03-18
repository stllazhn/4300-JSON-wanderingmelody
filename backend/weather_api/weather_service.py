from dotenv import load_dotenv
import requests
import os

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_current_weather(location):
    """
    Get current weather data for a given location
    
    Args:
        location (str): City name or location
        
    Returns:
        dict: Weather data or None if there was an error
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": location,
        "appid": API_KEY,
        "units": "metric"  # change to "imperial" if wanted
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # raise exception for 4XX/5XX responses
        
        data = response.json()
        
        # extract relevant weather information
        weather_info = {
            "location": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather_main": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "weather_icon": data["weather"][0]["icon"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "cloudiness": data["clouds"]["all"],
            "rain": data.get("rain", {}).get("1h", 0),
            "sunrise": data["sys"]["sunrise"],
            "sunset": data["sys"]["sunset"]
        }
        
        return weather_info
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None