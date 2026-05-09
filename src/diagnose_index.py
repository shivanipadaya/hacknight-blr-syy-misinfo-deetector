import argparse
import json

from rich.console import Console
from rich.table import Table

from src.elastic_client import get_elasticsearch_client
from src.settings import get_settings


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Elasticsearch index state.")
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    client = get_elasticsearch_client()
    index = settings.elasticsearch_index

    exists = client.indices.exists(index=index)
    console.print(f"[bold]Index:[/bold] {index}")
    console.print(f"[bold]Exists:[/bold] {exists}")
    if not exists:
        return

    count = client.count(index=index).get("count", 0)
    console.print(f"[bold]Document count:[/bold] {count}")

    try:
        pipeline = client.ingest.get_pipeline(id="elser_sparse_pipeline")
        console.print("[bold]Pipeline exists:[/bold] yes")
        console.print(json.dumps(pipeline.body if hasattr(pipeline, "body") else pipeline, indent=2)[:2000])
    except Exception as exc:
        console.print(f"[yellow]Pipeline missing/error:[/yellow] {exc}")

    response = client.search(
        index=index,
        body={
            "size": args.sample_size,
            "_source": ["url", "source_name", "source_type", "title", "content"],
            "query": {"match_all": {}},
        },
    )
    hits = response.get("hits", {}).get("hits", [])
    table = Table(title="Sample Documents")
    table.add_column("Score")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("URL")
    table.add_column("Content chars")
    for hit in hits:
        source = hit.get("_source", {})
        table.add_row(
            str(hit.get("_score", "")),
            source.get("source_name", ""),
            source.get("title", "")[:80],
            source.get("url", "")[:80],
            str(len(source.get("content", "") or "")),
        )
    console.print(table)


if __name__ == "__main__":
    main()
