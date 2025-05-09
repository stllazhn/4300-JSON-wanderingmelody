import re
import json
import numpy as np
import os
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def load_reviews_database(file_path=None):
    if file_path is None:
        # Get absolute path to the JSON file relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'datasets', 'countries_reviews.json')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reviews_database = json.load(file)
    return reviews_database

# Load the reviews database
REVIEWS_DATABASE = load_reviews_database()

def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.
    Note: for simplicity, lowercase everything. Do not remove duplicate words.
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.
    
    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    outcome = re.findall(r"[a-zA-Z]+", text.lower())
    return outcome

def find_similar_locations(country: str, mood: str) -> List[Tuple[str, float]]:
    """
    Finds the top 3 locations with reviews most similar to the user's mood input
    for a specific country.
    
    Parameters
    ----------
    country : str
        The country to search for locations in.
    mood : str
        The user's mood/preference input string.
        
    Returns
    -------
    List[Tuple[str, float]]
        A list of tuples containing (location, similarity_score) for the top 3 most similar reviews.
    """
    # Find the country data
    country_data = None
    for data in REVIEWS_DATABASE:
        if data["country"].lower() == country.lower():
            country_data = data
            break
    
    if country_data is None:
        return [("No locations found", 0.0), ("No locations found", 0.0), ("No locations found", 0.0)]
    
    # Extract all locations and reviews for this country
    locations_reviews = []
    i = 1
    while True:
        location = country_data.get(f"location_{i}", "")
        review = country_data.get(f"top_review_{i}", "")
        if not location or not review:
            break  # Stop when we don't find any more locations
        locations_reviews.append((location, review))
        i += 1
    
    # Extract just the review texts for vectorization
    review_texts = [review for _, review in locations_reviews]
    
    # Add the user input to the list of texts to vectorize
    all_texts = [mood] + review_texts
    
    # Tokenize and create document-term matrix
    tokenized_texts = [' '.join(tokenize(text)) for text in all_texts]
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(tokenized_texts)
    
    # Calculate cosine similarity between user input and all reviews
    user_vector = dtm[0:1]
    review_vectors = dtm[1:]
    similarities = cosine_similarity(user_vector, review_vectors).flatten()
    
    # Create a list of (location, similarity) tuples
    similarity_results = [
        (locations_reviews[i][0], float(similarities[i]))
        for i in range(len(similarities))
    ]
    
    # Sort by similarity score in descending order
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    
    # If we have fewer than 3 results, pad with dummy entries
    while len(similarity_results) < 3:
        similarity_results.append(("No location found", 0.0))
    
    # Return the top 3 results
    return similarity_results[:3]

# Test the function
def test_similarity_search():
    """
    Test the similarity search function with different mood inputs.
    """
    print("Test 1: China, I love historic monuments with amazing architecture")
    results = find_similar_locations("China", "I love historic monuments with amazing architecture")
    print_results(results)
    
    print("\nTest 2: China, I want to see breathtaking views")
    results = find_similar_locations("China", "I want to see breathtaking views")
    print_results(results)
    
    print("\nTest 3: China, tree")
    results = find_similar_locations("China", "tree")
    print_results(results)
    
    print("\nTest 4: China, Beautiful castles")
    results = find_similar_locations("China", "Beautiful castles")
    print_results(results)

def print_results(results):
    """Helper function to print the results."""
    for location, score in results:
        print(f"{location}: Similarity score = {score:.4f}")

if __name__ == "__main__":
    test_similarity_search()