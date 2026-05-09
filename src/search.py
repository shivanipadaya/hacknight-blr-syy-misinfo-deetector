from elasticsearch import Elasticsearch

from src.elastic_client import get_elasticsearch_client
from src.models import SearchHit
from src.settings import get_settings


def _snippet(source: dict, max_chars: int = 900) -> str:
    content = source.get("content") or ""
    content = " ".join(content.split())
    return content[:max_chars]


def _elser_clause(query: str) -> dict:
    settings = get_settings()
    if not settings.elser_inference_id:
        raise RuntimeError("Set ELSER_INFERENCE_ID to your Elastic Inference Service ELSER endpoint ID.")
    if settings.elser_query_mode == "text_expansion":
        return {
            "text_expansion": {
                "content_embedding": {
                    "model_id": settings.elser_inference_id,
                    "model_text": query,
                }
            }
        }

    return {
        "sparse_vector": {
            "field": "content_embedding",
            "inference_id": settings.elser_inference_id,
            "query": query,
        }
    }


def build_hybrid_query(claim: str, size: int) -> dict:
    return {
        "size": size,
        "track_total_hits": False,
        "_source": [
            "url",
            "source_name",
            "source_type",
            "title",
            "content",
            "published_at",
            "crawled_at",
        ],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": claim,
                            "fields": ["title^3", "content", "source_name^2"],
                            "type": "best_fields",
                        }
                    },
                    _elser_clause(claim),
                ],
                "minimum_should_match": 1,
            }
        },
    }


def search_claim(
    claim: str,
    *,
    size: int = 30,
    client: Elasticsearch | None = None,
) -> list[SearchHit]:
    settings = get_settings()
    client = client or get_elasticsearch_client()
    response = client.search(index=settings.elasticsearch_index, body=build_hybrid_query(claim, size))

    hits: list[SearchHit] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        hits.append(
            SearchHit(
                doc_id=hit.get("_id", ""),
                url=source.get("url", ""),
                title=source.get("title", ""),
                source_name=source.get("source_name", "unknown"),
                source_type=source.get("source_type", "unknown"),
                published_at=source.get("published_at"),
                snippet=_snippet(source),
                score=float(hit.get("_score") or 0.0),
            )
        )
    return hits
