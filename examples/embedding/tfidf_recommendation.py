import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def clean_text(text):
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def recommendations(name, top_k=10):
    csv_path = Path(__file__).parent / "data" / "Seattle_Hotels.csv"
    df = pd.read_csv(csv_path, encoding="latin-1")
    df["desc_clean"] = df["desc"].fillna("").map(clean_text)

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(df["desc_clean"])
    # TF-IDF 向量已做过归一化，这样算出的分数等价于余弦相似度。
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    hotel_to_index = pd.Series(df.index, index=df["name"])
    hotel_index = hotel_to_index[name]
    scores = list(enumerate(cosine_similarities[hotel_index]))
    scores.sort(key=lambda item: item[1], reverse=True)
    result_indexes = [index for index, _ in scores if index != hotel_index][:top_k]
    return df.loc[result_indexes, ["name", "address"]]


if __name__ == "__main__":
    print(recommendations("Hilton Seattle Airport & Conference Center"))
