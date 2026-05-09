import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from rich.console import Console

from src.elastic_client import get_elasticsearch_client
from src.settings import ROOT_DIR, get_settings


console = Console()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_pipeline(client: Elasticsearch) -> None:
    settings = get_settings()
    if not settings.elser_inference_id:
        raise RuntimeError("Set ELSER_INFERENCE_ID to your Elastic Inference Service ELSER endpoint ID.")
    pipeline = load_json(ROOT_DIR / "config" / "elser_pipeline.json")
    pipeline["processors"][0]["inference"]["model_id"] = settings.elser_inference_id
    client.ingest.put_pipeline(id="elser_sparse_pipeline", body=pipeline)
    console.print(
        f"Created ingest pipeline: elser_sparse_pipeline using {settings.elser_inference_id}"
    )


def create_index(client: Elasticsearch, recreate: bool = False) -> None:
    settings = get_settings()
    index_name = settings.elasticsearch_index
    mapping = load_json(ROOT_DIR / "config" / "elastic_index.json")

    if client.indices.exists(index=index_name):
        if not recreate:
            console.print(f"[yellow]Index already exists: {index_name}[/yellow]")
            return
        client.indices.delete(index=index_name)
        console.print(f"Deleted existing index: {index_name}")

    client.indices.create(index=index_name, body=mapping)
    console.print(f"Created index: {index_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Elasticsearch for claim checking.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the index.")
    parser.add_argument(
        "--skip-inference-endpoint",
        action="store_true",
        help="Deprecated; inference endpoints are expected to exist in Elastic Inference Service.",
    )
    args = parser.parse_args()

    client = get_elasticsearch_client()
    create_pipeline(client)
    create_index(client, recreate=args.recreate)


if __name__ == "__main__":
    main()
