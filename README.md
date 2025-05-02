# 🎶 WanderingMelody

A music recommendation engine that helps you explore the world through sound. Simply describe your mood, pick a country, share your age and weather—WanderingMelody returns a personalized soundtrack that mirrors the emotional and cultural essence of a place.

🟢 [Live App](http://4300showcase.infosci.cornell.edu:5245/)

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Features](#features)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## 🧠 About the Project

**WanderingMelody** was designed to simulate a musical journey around the world. You tell us how you feel, and we recommend music that matches that vibe—factoring in your age, genre preference, country, and weather.

> ❌ **NOTE:** Originally, this app planned to fetch real-time weather data via APIs. However, due to server restrictions, this feature was removed. Instead, users select their current weather condition from a dropdown, which the system then maps to a corresponding emotional tone. This still preserves the mood-based filtering while maintaining compatibility with hosting limitations:contentReference[oaicite:0]{index=0}.

---

## 🛠 Built With

- **Python (Flask)** – Backend server
- **HTML/CSS/JS** – Frontend interface with dark/light themes
- **NLTK** – Tokenization, sentiment analysis, WordNet synonyms/antonyms
- **Scikit-learn** – TF-IDF, SVD for latent topic modeling
- **Pandas & NumPy** – Data handling
- **Spotify Lyrics Dataset** – Primary song database
- **Google Maps Review Data** – For location similarity

---

## 🚀 Features

- 🎧 **Mood-Based Filtering** – Matches songs based on your emotional description.
- 🌤 **Weather-Aware Scoring** – Selectable weather input (e.g., rainy = mellow music).
- 🎂 **Age-Based Sentiment** – Tailors songs to age-influenced sentiment profiles.
- 🧠 **Latent Topic Modeling** – Uses SVD to compare lyric themes and user input.
- 🚫 **Antonym Penalty** – Avoids songs with opposite meanings to your input.
- 🔍 **Word Match Boost** – Bonuses for direct lyric overlap with your input.
- 🌎 **Location-Based Exploration** – Suggests locations based on mood similarity using Google Reviews.
- 🌟 **Popularity Ranking** – Spotify’s popularity score affects song ranking.
- 🎯 **Transparent Scoring** – See *why* each song was recommended.

---

## 🔍 How It Works

1. **Input**: User submits a mood description, age, genre, country, and weather type.
2. **Preprocessing**: Input is tokenized, stop words removed, and lemmatized.
3. **Theme Matching**: SVD compares user mood against lyric topic vectors.
4. **Filtering**: Age, sentiment, and antonym-based filters refine the pool.
5. **Scoring**: Final song scores combine topic match, word overlap, sentiment fit, and popularity.
6. **Output**: Top songs are returned along with an explanation for each choice.

---

## 🧰 Getting Started

### 🖥️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/WanderingMelody.git
   cd WanderingMelody

---

## 🙌 Acknowledgements

- **CS/INFO 4300 at Cornell University** – For the project framework and instruction
- **NLTK** – For natural language processing tools (tokenization, sentiment analysis, WordNet)
- **Spotify** – For song lyrics, popularity scores, and metadata
- **VANTA.js** – For animated background effects
- **OpenWeather** – (initially considered but ultimately not used due to deployment constraints)

