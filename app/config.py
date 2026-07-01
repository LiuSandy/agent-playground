import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    )
