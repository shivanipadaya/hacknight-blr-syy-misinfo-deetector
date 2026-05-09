import argparse
import json

from rich.console import Console
from rich.table import Table

from src.agent import generate_agent_answer
from src.injection_guard import detect_prompt_injection
from src.models import Citation, ClaimCheckResult, RelevantResource, SearchHit, Verdict
from src.rerank import rerank_claim
from src.search import search_claim


console = Console()


def _citation_from_hit(hit: SearchHit) -> Citation:
    return Citation(
        title=hit.title or hit.source_name,
        url=hit.url,
        source_name=hit.source_name,
        published_at=hit.published_at,
        quoted_snippet=hit.snippet,
        relevance_score=hit.rerank_score if hit.rerank_score is not None else hit.score,
    )


def _resource_from_hit(hit: SearchHit) -> RelevantResource:
    relevance_score = hit.rerank_score if hit.rerank_score is not None else hit.score
    return RelevantResource(
        resource=hit.url,
        data={
            "title": hit.title or hit.source_name,
            "snippet": hit.snippet,
        },
        metadata={
            "doc_id": hit.doc_id,
            "source_name": hit.source_name,
            "source_type": hit.source_type,
            "published_at": hit.published_at,
            "retrieval_score": hit.score,
            "rerank_score": hit.rerank_score,
            "relevance_score": relevance_score,
        },
    )


def _simple_verdict(claim: str, evidence: list[SearchHit]) -> tuple[Verdict, str]:
    if not evidence:
        return (
            Verdict.insufficient_evidence,
            "No reliable indexed source matched the claim strongly enough.",
        )

    official_hits = [hit for hit in evidence if hit.source_type in {"official", "official_social"}]
    best = official_hits[0] if official_hits else evidence[0]

    return (
        Verdict.partially_supported,
        (
            "Relevant source material was found, but this retrieval layer does not make a final "
            f"truth judgment without the answer-generation layer. Strongest source: {best.source_name}."
        ),
    )


def check_claim(claim: str, *, retrieval_size: int = 30, top_n: int = 5) -> ClaimCheckResult:
    injection_warnings = detect_prompt_injection(claim)
    hits = search_claim(claim, size=retrieval_size)
    for hit in hits:
        injection_warnings.extend(detect_prompt_injection(hit.snippet))

    reranked = rerank_claim(claim, hits, top_n=top_n)
    try:
        verdict, explanation, agent_answer = generate_agent_answer(claim, reranked)
    except Exception as exc:
        verdict, explanation = _simple_verdict(claim, reranked)
        agent_answer = f"Elastic LLM inference failed: {exc}"

    return ClaimCheckResult(
        claim=claim,
        verdict=verdict,
        explanation=explanation,
        agent_answer=agent_answer,
        resources=[_resource_from_hit(hit) for hit in reranked],
        citations=[_citation_from_hit(hit) for hit in reranked],
        injection_warnings=injection_warnings,
    )


def print_result(result: ClaimCheckResult) -> None:
    console.print(f"[bold]Claim:[/bold] {result.claim}")
    console.print(f"[bold]Verdict:[/bold] {result.verdict}")
    console.print(f"[bold]Explanation:[/bold] {result.explanation}")

    table = Table(title="Citations")
    table.add_column("Score")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("URL")
    for citation in result.citations:
        table.add_row(
            f"{citation.relevance_score:.3f}",
            citation.source_name,
            citation.title[:80],
            citation.url,
        )
    console.print(table)

    if result.injection_warnings:
        console.print("[yellow]Injection warnings detected:[/yellow]")
        for warning in sorted(set(result.injection_warnings)):
            console.print(f"- {warning}")

    console.print("\n[bold]Use --json to print full resource/data/metadata output.[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a Bengaluru misinformation claim.")
    parser.add_argument("claim", help="The viral claim to check.")
    parser.add_argument("--retrieval-size", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print full JSON response.")
    args = parser.parse_args()

    result = check_claim(args.claim, retrieval_size=args.retrieval_size, top_n=args.top_n)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
