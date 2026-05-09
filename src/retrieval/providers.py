from datetime import UTC, datetime

from elasticsearch import Elasticsearch

from src.elastic_client import get_elasticsearch_client
from src.rerank import rerank_claim
from src.schemas.verification import SourceEnvelope
from src.search import search_claim


class ElasticsearchRetrievalProvider:
    def __init__(self, client: Elasticsearch | None = None) -> None:
        self.client = client or get_elasticsearch_client()

    def provide_sources(self, claim: str, *, retrieval_size: int = 30, top_k: int = 5) -> list[SourceEnvelope]:
        hits = search_claim(claim, size=retrieval_size, client=self.client)
        reranked = rerank_claim(claim, hits, top_n=top_k, client=self.client)
        today = datetime.now(UTC).date().isoformat()
        sources: list[SourceEnvelope] = []
        for index, hit in enumerate(reranked, start=1):
            sources.append(
                SourceEnvelope(
                    id=hit.doc_id or f"doc_{index}",
                    url=hit.url,
                    text=hit.snippet,
                    date_of_scraping=hit.published_at or today,
                    title=hit.title,
                    source_name=hit.source_name,
                    source_type=hit.source_type,
                    retrieval_score=hit.score,
                    rerank_score=hit.rerank_score,
                )
            )
        return sources
