import argparse
import hashlib
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track

from src.crawler_config import load_sources
from src.elastic_client import get_elasticsearch_client
from src.models import SourceDocument
from src.settings import get_settings


console = Console()


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = " ".join(data.split())
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def normalize_url(url: str) -> str:
    clean_url, _fragment = urldefrag(url)
    return clean_url.rstrip("/")


def same_domain(url: str, seed_url: str) -> bool:
    return urlparse(url).netloc == urlparse(seed_url).netloc


def fetch_html(url: str, timeout: int = 20) -> str | None:
    response = requests.get(
        url,
        headers={
            "User-Agent": "blr-misinfo-detector/0.1 (+https://example.local)",
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=timeout,
    )
    content_type = response.headers.get("content-type", "")
    if response.status_code >= 400 or "html" not in content_type:
        return None
    return response.text


def extract_title_and_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    extractor = TextExtractor()
    extractor.feed(html)
    return title, extractor.text()


def extract_links(html: str, base_url: str, seed_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if same_domain(absolute, seed_url):
            links.append(absolute)
    return links


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_document(doc: SourceDocument) -> None:
    settings = get_settings()
    client = get_elasticsearch_client()
    doc_id = doc.content_hash or content_hash(doc.content)
    client.index(
        index=settings.elasticsearch_index,
        id=doc_id,
        document=doc.model_dump(mode="json"),
        pipeline="elser_sparse_pipeline",
    )


def crawl_source(source: dict, *, max_pages: int, depth: int, min_chars: int) -> int:
    seed_url = normalize_url(source["url"])
    queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
    visited: set[str] = set()
    indexed = 0

    while queue and indexed < max_pages:
        url, current_depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch_html(url)
        except requests.RequestException as exc:
            console.print(f"[yellow]Fetch failed:[/yellow] {url} ({exc})")
            continue

        if not html:
            continue

        title, text = extract_title_and_text(html)
        if len(text) >= min_chars:
            hash_value = content_hash(text)
            doc = SourceDocument(
                url=url,
                source_name=source["name"],
                source_type=source["type"],
                title=title,
                content=text,
                crawled_at=datetime.now(timezone.utc),
                content_hash=hash_value,
            )
            index_document(doc)
            indexed += 1
            console.print(f"Indexed [{source['name']}]: {title or url}")

        if current_depth < depth:
            for link in extract_links(html, url, seed_url):
                if link not in visited:
                    queue.append((link, current_depth + 1))

    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Elasticsearch from trusted web sources.")
    parser.add_argument("--max-pages-per-source", type=int, default=5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    total = 0
    sources = load_sources()
    for source in track(sources, description="Ingesting trusted sources"):
        total += crawl_source(
            source,
            max_pages=args.max_pages_per_source,
            depth=args.depth,
            min_chars=args.min_chars,
        )

    console.print(f"[bold]Indexed {total} documents.[/bold]")


if __name__ == "__main__":
    main()
