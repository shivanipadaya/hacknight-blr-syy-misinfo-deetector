import logging

from elasticsearch import ApiError
from elasticsearch import Elasticsearch

from src.elastic_client import ELASTIC_JSON_HEADERS, get_elasticsearch_client
from src.models import SearchHit
from src.settings import get_settings

logger = logging.getLogger(__name__)


def _document_text(hit: SearchHit) -> str:
    return (
        f"Title: {hit.title}\n"
        f"Source: {hit.source_name} ({hit.source_type})\n"
        f"URL: {hit.url}\n"
        f"Content: {hit.snippet}"
    )


def _apply_rerank_results(hits: list[SearchHit], results: list[dict], top_n: int) -> list[SearchHit]:
    reranked: list[SearchHit] = []
    for result in results:
        index = result.get("index")
        if index is None or int(index) >= len(hits):
            continue
        hit = hits[int(index)].model_copy()
        hit.rerank_score = float(result.get("relevance_score") or 0.0)
        reranked.append(hit)
    return reranked[:top_n]


def rerank_with_elastic(
    claim: str,
    hits: list[SearchHit],
    *,
    top_n: int,
    client: Elasticsearch | None = None,
) -> list[SearchHit]:
    settings = get_settings()
    client = client or get_elasticsearch_client()
    payload = {
        "query": claim,
        "input": [_document_text(hit) for hit in hits],
        "return_documents": False,
        "top_n": min(top_n, len(hits)),
    }
    request_client = client.options(headers=ELASTIC_JSON_HEADERS)
    response = request_client.perform_request(
        "POST",
        f"/_inference/rerank/{settings.jina_rerank_inference_id}",
        body=payload,
    )
    body = response.body if hasattr(response, "body") else response
    return _apply_rerank_results(hits, body.get("rerank", []), top_n)


def rerank_claim(
    claim: str,
    hits: list[SearchHit],
    *,
    top_n: int = 5,
    client: Elasticsearch | None = None,
) -> list[SearchHit]:
    if not hits:
        return []

    settings = get_settings()
    if not settings.rerank_enabled:
        return hits[:top_n]

    try:
        return rerank_with_elastic(claim, hits, top_n=top_n, client=client)
    except ApiError as exc:
        logger.warning(
            "rerank_failed_falling_back_to_retrieval_scores",
            extra={
                "status_code": getattr(exc, "status_code", None),
                "error": str(exc),
                "endpoint": settings.jina_rerank_inference_id,
            },
        )
        return hits[:top_n]
