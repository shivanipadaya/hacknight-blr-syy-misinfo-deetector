from elasticsearch import Elasticsearch

from src.settings import get_settings


ELASTIC_JSON_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def get_elasticsearch_client() -> Elasticsearch:
    settings = get_settings()

    if settings.elastic_cloud_id:
        kwargs = {"cloud_id": settings.elastic_cloud_id}
    elif settings.elasticsearch_url:
        kwargs = {"hosts": [settings.elasticsearch_url]}
    else:
        raise RuntimeError(
            "Set ELASTIC_CLOUD_ID or ELASTICSEARCH_URL. The Kibana URL is not the Elasticsearch endpoint."
        )

    if settings.elasticsearch_api_key:
        kwargs["api_key"] = settings.elasticsearch_api_key
    elif settings.elasticsearch_username and settings.elasticsearch_password:
        kwargs["basic_auth"] = (
            settings.elasticsearch_username,
            settings.elasticsearch_password,
        )
    else:
        raise RuntimeError("Set ELASTICSEARCH_API_KEY or ELASTICSEARCH_USERNAME/PASSWORD.")

    return Elasticsearch(**kwargs)
