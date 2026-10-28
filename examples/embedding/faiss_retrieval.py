import os

import faiss
import numpy as np
from openai import OpenAI

DIMENSION = 1024

documents = [
    {
        "text": "迪士尼门票原则上不予退换，特殊情况下可按官方指引退款。",
        "metadata": {"source": "official_faq.pdf", "category": "退票政策"},
    },
    {
        "text": "奇妙年卡用户可在有效期内多次入园并享受部分折扣。",
        "metadata": {"source": "annual_pass.docx", "category": "会员权益"},
    },
    {
        "text": "在线购买的门票如需退款，须提前48小时通过原渠道申请。",
        "metadata": {"source": "online_policy.html", "category": "退票政策"},
    },
    {
        "text": "游客可以乘坐地铁到达迪士尼乐园。",
        "metadata": {"source": "transport.md", "category": "交通指南"},
    },
]


def embed(text, client):
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=DIMENSION,
        encoding_format="float",
    )
    return response.data[0].embedding


def search_documents(query, top_k=3):
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # Metadata 保存在 FAISS 之外，整数键同时作为向量 ID。
    metadata_store = {index: document for index, document in enumerate(documents)}
    vectors = np.asarray(
        [embed(document["text"], client) for document in documents],
        dtype="float32",
    )
    vector_ids = np.asarray(list(metadata_store.keys()), dtype="int64")

    # IndexFlatL2 执行精确的 L2 距离搜索，IndexIDMap 负责自定义 ID。
    index = faiss.IndexIDMap(faiss.IndexFlatL2(DIMENSION))
    index.add_with_ids(vectors, vector_ids)

    query_vector = np.asarray([embed(query, client)], dtype="float32")
    distances, result_ids = index.search(query_vector, k=top_k)

    for rank, (distance, result_id) in enumerate(
        zip(distances[0], result_ids[0]),
        start=1,
    ):
        if result_id == -1:
            continue
        document = metadata_store[int(result_id)]
        print(f"排名：{rank}，平方 L2 距离：{float(distance):.4f}")
        print("原文：", document["text"])
        print("Metadata：", document["metadata"])


if __name__ == "__main__":
    search_documents("我想了解迪士尼门票的退款流程")
