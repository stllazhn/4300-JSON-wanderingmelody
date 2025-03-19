# ml.py
import json
import os
import re
import pandas as pd
from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
from collections import Counter
import nltk
# from nltk.corpus import stopwords, wordnet
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from itertools import chain
import math

# Ensure NLTK resources are downloaded
# nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# Load datasets
spotify_df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
lyric_df = pd.read_csv("spotify_millsongdata.csv")



# Tokenization function
def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text."""
    return re.findall(r"[a-zA-Z]+", text.lower())

# Tokenize lyrics
def tokenize_lyrics(dict_of_lyrics, tokenize_method: Callable[[str], List[str]]):
    tokenized_lyrics = {}
    nrows = len(lyric_df)
    for i in range(nrows):
        lyrics = dict_of_lyrics[i]['text']
        tokenized_lyrics[i] = tokenize_method(lyrics)
    return tokenized_lyrics

# Build word-song count
def build_word_song_count(tokenize_method: Callable[[str], List[str]], tokenized_lyrics: Dict[int, List[str]]):
    song_count = {}
    for key in tokenized_lyrics:
        unique_tokens = set(token.casefold() for token in tokenized_lyrics[key])
        for token in unique_tokens:
            if token not in song_count:
                song_count[token] = set()
            song_count[token].add(key)
    return song_count

# Remove stop words
def remove_stop_words(list_of_stop_words, tokenized_lyrics):
    clean_stop_words = set()
    for word in list_of_stop_words:
        clean_stop_words.update(nltk.word_tokenize(word.lower()))
    stop_word_removed_lyrics = {}
    for i, lyrics in tokenized_lyrics.items():
        cleaned_lyrics = [word for word in lyrics if word.lower() not in clean_stop_words]
        stop_word_removed_lyrics[i] = cleaned_lyrics
    return stop_word_removed_lyrics

# Remove stop words from input
def remove_stop_words_input(tokenize, list_of_stop_words, input_words):
    list_tokens = tokenize(input_words)
    cleaned_words = [word for word in list_tokens if word not in list_of_stop_words]
    return cleaned_words

# Build inverted index
def build_inverted_index(songs: dict) -> dict:
    inverted_index = {}
    for song_num, tokens in songs.items():
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        for token, count in token_counts.items():
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((song_num, count))
    return inverted_index

# Compute IDF
def compute_idf(inv_idx, n_docs, min_df=0, max_df_ratio=1):
    return_dict = {}
    for word, word_list in inv_idx.items():
        word_count = len(word_list)
        if word_count >= min_df and word_count / n_docs <= max_df_ratio:
            idf = math.log2(n_docs/(1+word_count))
            return_dict[word] = idf
    return return_dict

# Get synonyms of a word
def get_synonyms_of_word(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name().replace("_", " ").lower())
    return set(synonyms)

# Get antonyms of a word
def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            if i.antonyms():
                antonyms.append(i.name().replace("_", " ").lower())
    return set(antonyms)

# Sort songs by polarity scores
def sort_polarity_scores(input_song_list, cleaned_tokenized_lyrics, polarity_type="compound"):
    sia = SentimentIntensityAnalyzer()
    dict_of_polarity_scores = {}
    for song_row_number in input_song_list:
        if song_row_number in cleaned_tokenized_lyrics:
            dict_of_polarity_scores[song_row_number] = sia.polarity_scores(" ".join(cleaned_tokenized_lyrics[song_row_number]))
    sorted_songs = sorted(dict_of_polarity_scores, key=lambda x: dict_of_polarity_scores[x][polarity_type], reverse=True)
    return sorted_songs

# Main function to process user input and recommend songs
def recommend_songs(user_genre_input, cleaned_tokenized_lyrics, clean_song_count):
    clean_genre_input = remove_stop_words(stopwords.words('english'), {0: tokenize(user_genre_input)})
    possible_songs_dict = {}
    for word in clean_genre_input[0]:
        if clean_song_count.get(word) is not None:
            possible_songs_dict[word] = clean_song_count[word]
    
    stemmed_user_input = []
    stemmed_user_input_preserve_original = []
    for word in clean_genre_input[0]:
        if possible_songs_dict.get(word) is None:
            stemmed_user_input_word = ps.stem(word)
            stemmed_user_input.append(stemmed_user_input_word)
            stemmed_user_input_preserve_original.append(stemmed_user_input_word)
        else:
            stemmed_user_input_preserve_original.append(word)
    
    for word in stemmed_user_input:
        if clean_song_count.get(word) is not None:
            possible_songs_dict[word] = clean_song_count[word]
    
    most_common_songs = []
    if possible_songs_dict:
        song_counts = Counter(song for song_list in possible_songs_dict.values() for song in song_list)
        max_number = max(song_counts.values(), default=0)
        most_common_songs = [song for song, count in song_counts.items() if count == max_number]
    
    word_synonym_dict = {}
    for word in stemmed_user_input_preserve_original:
        if possible_songs_dict.get(word) is None:
            set_of_synonyms = get_synonyms_of_word(word)
            word_synonym_dict[word] = list(set_of_synonyms)
    
    return most_common_songs, word_synonym_dict
