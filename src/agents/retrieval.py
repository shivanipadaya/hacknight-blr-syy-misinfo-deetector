import logging
from typing import Any

from src.schemas.verification import SourceEnvelope

logger = logging.getLogger(__name__)


class RetrievalAgent:
    def __init__(self, provider: Any | None = None) -> None:
        if provider is None:
            from src.retrieval.providers import ElasticsearchRetrievalProvider

            provider = ElasticsearchRetrievalProvider()
        self.provider = provider

    def run(self, query: str, *, retrieval_size: int, top_k: int) -> list[SourceEnvelope]:
        logger.info(
            "retrieval_agent_started",
            extra={"retrieval_size": retrieval_size, "top_k": top_k, "query_preview": query[:160]},
        )
        sources = self.provider.provide_sources(query, retrieval_size=retrieval_size, top_k=top_k)
        logger.info(
            "retrieval_agent_completed",
            extra={"source_count": len(sources), "source_ids": [source.id for source in sources]},
        )
        return sources
