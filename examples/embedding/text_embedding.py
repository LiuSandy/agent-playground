import os

from openai import OpenAI


def embed(text):
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024,
        encoding_format="float",
    )
    return response.data[0].embedding


if __name__ == "__main__":
    vector = embed("我想知道迪士尼的退票政策")
    print(len(vector))  # 1024
