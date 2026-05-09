import argparse
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console

from src.claim_check import check_claim, print_result
from src.crawler_config import load_sources
from src.ingest_seed_sources import content_hash, fetch_html, normalize_url, extract_title_and_text, index_document
from src.models import ClaimCheckResult, SourceDocument


console = Console()


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


def _allowed_domains() -> set[str]:
    domains: set[str] = set()
    for source in load_sources():
        domain = urlparse(source["url"]).netloc.lower()
        if domain:
            domains.add(domain)
    return domains


def _source_for_url(url: str) -> dict:
    parsed_domain = urlparse(url).netloc.lower()
    for source in load_sources():
        if urlparse(source["url"]).netloc.lower() == parsed_domain:
            return source
    return {
        "name": parsed_domain or "unknown",
        "type": "unknown",
        "url": url,
    }


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" not in parsed.netloc:
        return url
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def search_trusted_web(claim: str, *, max_results: int = 10) -> list[str]:
    domains = sorted(_allowed_domains())
    if not domains:
        return []

    site_filter = " OR ".join(f"site:{domain}" for domain in domains)
    query = f"{claim} ({site_filter})"
    response = requests.post(
        DUCKDUCKGO_HTML_URL,
        data={"q": query},
        headers={
            "User-Agent": "blr-misinfo-detector/0.1",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=25,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for link in soup.select("a.result__a, a.result-link"):
        href = link.get("href")
        if not href:
            continue
        url = normalize_url(_unwrap_duckduckgo_url(str(href)))
        domain = urlparse(url).netloc.lower()
        if domain not in domains or url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max_results:
            break
    return urls


def fallback_source_search_urls(claim: str, *, max_results: int = 10) -> list[str]:
    urls: list[str] = []
    for source in load_sources():
        base = source["url"].rstrip("/")
        domain = urlparse(base).netloc.lower()
        if not domain:
            continue
        if "deccanherald.com" in domain:
            urls.append(f"https://www.deccanherald.com/search?q={quote_plus(claim)}")
        elif "timesofindia.indiatimes.com" in domain:
            urls.append(f"https://timesofindia.indiatimes.com/topic/{quote_plus(claim)}")
        else:
            urls.append(base)
        if len(urls) >= max_results:
            break
    return urls


def ingest_url(url: str, *, min_chars: int = 500) -> bool:
    try:
        html = fetch_html(url)
    except requests.RequestException as exc:
        console.print(f"[yellow]Fetch failed:[/yellow] {url} ({exc})")
        return False

    if not html:
        return False

    title, text = extract_title_and_text(html)
    if len(text) < min_chars:
        console.print(f"[yellow]Skipping short page:[/yellow] {url}")
        return False

    source = _source_for_url(url)
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
    console.print(f"Indexed live result: {title or url}")
    return True


def live_check_claim(
    claim: str,
    *,
    search_results: int = 10,
    top_n: int = 5,
    min_chars: int = 500,
) -> ClaimCheckResult:
    try:
        urls = search_trusted_web(claim, max_results=search_results)
    except requests.RequestException as exc:
        console.print(f"[yellow]Web search failed, using source fallbacks:[/yellow] {exc}")
        urls = fallback_source_search_urls(claim, max_results=search_results)

    if not urls:
        urls = fallback_source_search_urls(claim, max_results=search_results)

    indexed = 0
    for url in urls:
        if ingest_url(url, min_chars=min_chars):
            indexed += 1

    console.print(f"[bold]Live indexed {indexed} pages before retrieval.[/bold]")
    return check_claim(claim, retrieval_size=max(30, search_results * 3), top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search trusted web sources, index them, then check a claim.")
    parser.add_argument("claim", help="The viral claim to check.")
    parser.add_argument("--search-results", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = live_check_claim(
        args.claim,
        search_results=args.search_results,
        top_n=args.top_n,
        min_chars=args.min_chars,
    )
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
