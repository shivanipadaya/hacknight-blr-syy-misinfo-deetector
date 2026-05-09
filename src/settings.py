from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    elastic_cloud_id: str | None = Field(default=None, alias="ELASTIC_CLOUD_ID")
    elasticsearch_url: str | None = Field(default=None, alias="ELASTICSEARCH_URL")
    elasticsearch_api_key: str | None = Field(default=None, alias="ELASTICSEARCH_API_KEY")
    elasticsearch_username: str | None = Field(default=None, alias="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str | None = Field(default=None, alias="ELASTICSEARCH_PASSWORD")
    elasticsearch_index: str = Field(default="blr_claim_sources_v1", alias="ELASTICSEARCH_INDEX")

    elser_inference_id: str = Field(default="", alias="ELSER_INFERENCE_ID")
    elser_query_mode: str = Field(default="sparse_vector", alias="ELSER_QUERY_MODE")

    jina_rerank_inference_id: str = Field(
        default="jina-reranker-v3",
        alias="JINA_RERANK_INFERENCE_ID",
    )
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    llm_inference_id: str = Field(
        default=".gp-llm-v2-chat_completion",
        alias="LLM_INFERENCE_ID",
    )
    llm_task_type: str = Field(default="chat_completion", alias="LLM_TASK_TYPE")
    llm_max_tokens: int = Field(default=900, alias="LLM_MAX_TOKENS")

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
