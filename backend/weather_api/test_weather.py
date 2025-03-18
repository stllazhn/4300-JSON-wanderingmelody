from weather_service import get_current_weather

# testing w london for now
def test_weather_service():
    location = "London"
    weather_data = get_current_weather(location)
    
    if weather_data:
        print("weather data retrieved successfully!!!!")
        print(f"location: {weather_data['location']}, {weather_data['country']}")
        print(f"temperature: {weather_data['temperature']}°C")
        print(f"weather: {weather_data['weather_description']}")
        print(f"wind: {weather_data['wind_speed']} m/s")
    else:
        print("failed to retrieve weather data :(")

if __name__ == "__main__":
    test_weather_service()