import pandas as pd

df = pd.read_json('spotify_millsongdata.json')
word_counts = df['text'].apply(lambda x: len(str(x).split()))

median_word_count = int(word_counts.median())

def truncate_words(text, word_limit):
    words = str(text).split()
    if len(words) <= word_limit:
        return text
    return ' '.join(words[:word_limit])

df['text'] = df['text'].apply(lambda x: truncate_words(x, median_word_count))
df.to_json('spotify_millsongdata_trunc.json', orient='records')