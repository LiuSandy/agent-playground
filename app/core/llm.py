from langchain_openai import ChatOpenAI

from app.config import Settings


def create_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )
