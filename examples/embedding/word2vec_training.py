from pathlib import Path

import jieba
from gensim.models import Word2Vec


def train_word2vec():
    corpus_path = Path(__file__).parent / "data" / "journey_to_the_west.txt"
    text = corpus_path.read_text(encoding="gb18030")

    sentences = []
    for paragraph in text.splitlines():
        words = [word.strip() for word in jieba.cut(paragraph) if word.strip()]
        if words:
            sentences.append(words)

    model = Word2Vec(
        sentences=sentences,
        vector_size=128,
        window=5,
        min_count=5,
        workers=4,
    )
    model.save("word2vec.model")
    return model


if __name__ == "__main__":
    model = train_word2vec()
    print(model.wv.similarity("孙悟空", "猪八戒"))
    print(model.wv.similarity("孙悟空", "孙行者"))
    print(model.wv.most_similar("孙悟空"))
