from sklearn.feature_extraction.text import CountVectorizer


def show_count_vectors():
    corpus = [
        "this hotel provides airport shuttle service",
        "this hotel has free breakfast and airport transportation",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names_out())
    print(count_matrix.toarray())


if __name__ == "__main__":
    show_count_vectors()
