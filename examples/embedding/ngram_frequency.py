from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_words(corpus, n=1, k=20):
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(corpus.fillna(""))
    frequencies = matrix.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    pairs = zip(features, frequencies)
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:k]


def show_hotel_ngrams():
    csv_path = Path(__file__).parent / "data" / "Seattle_Hotels.csv"
    df = pd.read_csv(csv_path, encoding="latin-1")
    print("Unigram:", get_top_n_words(df["desc"], n=1))
    print("Bigram:", get_top_n_words(df["desc"], n=2))
    print("Trigram:", get_top_n_words(df["desc"], n=3))


if __name__ == "__main__":
    show_hotel_ngrams()
