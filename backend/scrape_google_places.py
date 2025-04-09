import time
import random
import argparse
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
from difflib import SequenceMatcher

def similar(a, b):
    """Calculate string similarity ratio"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_duplicate(place, places_list, threshold=0.7):
    """Check if a place is a duplicate of any in the list based on name similarity"""
    for existing_place in places_list:
        if similar(place['name'], existing_place['name']) > threshold:
            return True
    return False

def get_google_maps_recommendations(city_name):
    """
    Scrape recommended places from Google Maps for a given city using Selenium
    
    Args:
        city_name (str): Name of the city to search for
    
    Returns:
        list: List of dictionaries containing place information
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=3")  # Suppress most console messages
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Suppress Chrome driver logs
    from selenium.webdriver.chrome.service import Service
    service = Service(log_path=os.devnull)
    
    driver = webdriver.Chrome(options=chrome_options, service=service)
    wait = WebDriverWait(driver, 10)
    all_places = []  # Store all places before filtering
    unique_places = []  # Store only unique places
    
    try:
        # Format search query and go to Google Maps
        query = f"best places to visit in {city_name}"
        driver.get("https://www.google.com/maps")
        
        # Accept cookies if prompted (may vary by region)
        try:
            cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all')]")))
            cookie_button.click()
            time.sleep(1)
        except (TimeoutException, NoSuchElementException):
            pass  # No cookie prompt appeared
        
        # Find search box and enter query
        search_box = wait.until(EC.presence_of_element_located((By.ID, "searchboxinput")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)
        
        # Wait for results to load
        time.sleep(5)
        
        # Scroll to load more results
        for _ in range(4):  # Scroll more to get more places
            try:
                driver.execute_script("document.querySelector('div[role=\"feed\"]').scrollTop += 500")
                time.sleep(1)
            except:
                break
        
        # Find place elements
        place_elements = driver.find_elements(By.CSS_SELECTOR, "div.Nv2PK")
        
        if not place_elements:
            # Try alternative selector
            place_elements = driver.find_elements(By.CSS_SELECTOR, "div.THOPZb")
        
        if not place_elements:
            # Try another alternative selector
            place_elements = driver.find_elements(By.CSS_SELECTOR, "a[href^='/maps/place']")
        
        # Process places
        for element in place_elements[:30]:  # Get more than we need to find highest rated
            place_info = {}
            
            try:
                # Extract place name
                name_element = element.find_element(By.CSS_SELECTOR, "div.qBF1Pd")
                place_info['name'] = name_element.text.strip()
            except NoSuchElementException:
                try:
                    # Alternative name selector
                    name_element = element.find_element(By.CSS_SELECTOR, "div.fontHeadlineSmall")
                    place_info['name'] = name_element.text.strip()
                except NoSuchElementException:
                    continue  # Skip if we can't find the name
            
            try:
                # Extract rating
                rating_element = element.find_element(By.CSS_SELECTOR, "span.MW4etd")
                rating_text = rating_element.text.strip()
                place_info['rating'] = float(rating_text)
            except (NoSuchElementException, ValueError):
                place_info['rating'] = 0.0  # Default rating if not found or can't convert
            
            try:
                # Extract number of reviews
                reviews_element = element.find_element(By.CSS_SELECTOR, "span.UY7F9")
                reviews_text = reviews_element.text.strip('()')
                reviews_text = reviews_text.replace(',', '')  # Remove commas
                place_info['reviews'] = int(reviews_text)
            except (NoSuchElementException, ValueError):
                place_info['reviews'] = 0  # Default if not found
            
            try:
                # Extract category/address info (usually the second line)
                detail_elements = element.find_elements(By.CSS_SELECTOR, "div.W4Efsd > span.W4Efsd")
                if detail_elements and len(detail_elements) > 0:
                    place_info['category'] = detail_elements[0].text.strip()
                if detail_elements and len(detail_elements) > 1:
                    place_info['address'] = detail_elements[1].text.strip()
            except NoSuchElementException:
                pass  # Details not available
            
            # Add to our results
            if place_info.get('name'):
                all_places.append(place_info)
            
            # Respect rate limits
            time.sleep(random.uniform(0.2, 0.5))
        
        # Try a different search if needed
        if len(all_places) < 5:
            # Search for "attractions in [city]" instead
            driver.get("https://www.google.com/maps")
            time.sleep(2)
            
            search_box = wait.until(EC.presence_of_element_located((By.ID, "searchboxinput")))
            search_box.clear()
            search_box.send_keys(f"attractions in {city_name}")
            search_box.send_keys(Keys.ENTER)
            
            time.sleep(5)
            
            # Try to find results with this new query
            place_elements = driver.find_elements(By.CSS_SELECTOR, "div.Nv2PK, div.THOPZb")
            
            for element in place_elements[:30]:
                place_info = {}
                try:
                    name = element.find_element(By.CSS_SELECTOR, "div.fontHeadlineSmall, div.qBF1Pd").text.strip()
                    place_info['name'] = name
                    
                    # Try to get rating
                    try:
                        rating_element = element.find_element(By.CSS_SELECTOR, "span.MW4etd")
                        rating_text = rating_element.text.strip()
                        place_info['rating'] = float(rating_text)
                    except (NoSuchElementException, ValueError):
                        place_info['rating'] = 0.0
                        
                    all_places.append(place_info)
                except:
                    continue
        
        # Sort places by rating (highest first)
        all_places.sort(key=lambda x: (x.get('rating', 0), x.get('reviews', 0)), reverse=True)
        
        # Filter out duplicates while preserving order
        for place in all_places:
            if not is_duplicate(place, unique_places):
                unique_places.append(place)
                if len(unique_places) >= 5:  # Stop after we have 5 unique places
                    break
        
        return unique_places
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
    finally:
        driver.quit()

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Get top 5 unique rated places from Google Maps for a city')
    parser.add_argument('city', type=str, help='Name of the city to search for')
    args = parser.parse_args()
    
    print(f"Searching for top rated places in {args.city}...")
    recommendations = get_google_maps_recommendations(args.city)
    
    if recommendations:
        print(f"\nTop {len(recommendations)} highest rated unique places in {args.city}:\n")
        for i, place in enumerate(recommendations, 1):
            print(f"{i}. {place.get('name', 'Unknown')}")
            if 'rating' in place:
                stars = '★' * int(place['rating']) + '☆' * (5 - int(place['rating']))
                print(f"   Rating: {place['rating']} {stars}")
            if 'reviews' in place and place['reviews'] > 0:
                print(f"   Reviews: {place['reviews']:,}")  # Format with commas
            if 'category' in place:
                print(f"   Category: {place['category']}")
            if 'address' in place:
                print(f"   Address: {place['address']}")
            print()
    else:
        print(f"No recommendations found for {args.city}. Try a different search query.")

if __name__ == "__main__":
    main()