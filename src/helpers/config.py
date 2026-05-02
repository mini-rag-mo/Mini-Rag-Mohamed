from pydantic_settings import BaseSettings, ConfigDict
from typing import List

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: List[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    POSTGRES_URL: str | None = None

    GENERATION_BACKEND: str | None = None
    EMBEDDING_BACKEND: str | None = None

    OPENAI_API_KEY: str | None = None
    OPENAI_API_URL: str | None = None
    COHERE_API_KEY: str | None = None

    GENERATION_MODEL_ID_LITERAL: List[str] | None = None
    GENERATION_MODEL_ID: str | None = None
    EMBEDDING_MODEL_ID: str | None = None
    EMBEDDING_MODEL_SIZE: int | None = None
    INPUT_DAFAULT_MAX_CHARACTERS: int | None = None
    GENERATION_DAFAULT_MAX_TOKENS: int | None = None
    GENERATION_DAFAULT_TEMPERATURE: float | None = None

    VECTOR_DB_BACKEND_LITERAL: List[str] | None = None
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str | None = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    model_config = ConfigDict(env_file=".env")

def get_settings():
    return Settings()