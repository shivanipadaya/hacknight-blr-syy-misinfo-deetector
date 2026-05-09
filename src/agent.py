import json

from elasticsearch import Elasticsearch

from src.elastic_client import ELASTIC_JSON_HEADERS, get_elasticsearch_client
from src.injection_guard import build_answer_prompt
from src.models import SearchHit, Verdict
from src.settings import get_settings


def _evidence_block(hits: list[SearchHit]) -> str:
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] title: {hit.title or hit.source_name}",
                    f"[{idx}] source: {hit.source_name}",
                    f"[{idx}] source_type: {hit.source_type}",
                    f"[{idx}] url: {hit.url}",
                    f"[{idx}] published_at: {hit.published_at}",
                    f"[{idx}] relevance_score: {score}",
                    f"[{idx}] snippet: {hit.snippet}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_grounded_prompt(claim: str, hits: list[SearchHit]) -> str:
    schema = {
        "verdict": "SUPPORTED | CONTRADICTED | PARTIALLY_SUPPORTED | INSUFFICIENT_EVIDENCE",
        "explanation": "short explanation grounded only in evidence",
        "used_citation_numbers": [1],
    }
    evidence = _evidence_block(hits)
    prompt = build_answer_prompt(claim, evidence)
    return (
        f"{prompt}\n\n"
        "Return strict JSON only. Do not include markdown.\n"
        f"JSON schema example: {json.dumps(schema)}"
    )


def _parse_completion_response(body: dict) -> str:
    completions = body.get("completion") or []
    if not completions:
        return ""
    return completions[0].get("result", "")


def _parse_chat_completion_response(body: dict) -> str:
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "".join(parts)
    return body.get("content") or body.get("result") or ""


def _parse_agent_json(raw_text: str) -> tuple[Verdict, str]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return Verdict.insufficient_evidence, raw_text.strip() or "The LLM endpoint returned no answer."

    verdict_text = str(data.get("verdict", Verdict.insufficient_evidence)).upper()
    try:
        verdict = Verdict(verdict_text)
    except ValueError:
        verdict = Verdict.insufficient_evidence

    explanation = str(data.get("explanation") or "").strip()
    if not explanation:
        explanation = "The LLM endpoint did not provide an explanation."
    return verdict, explanation


def generate_agent_answer(
    claim: str,
    hits: list[SearchHit],
    *,
    client: Elasticsearch | None = None,
) -> tuple[Verdict, str, str]:
    if not hits:
        return (
            Verdict.insufficient_evidence,
            "No indexed source documents were retrieved for this claim.",
            "",
        )

    settings = get_settings()
    client = client or get_elasticsearch_client()
    request_client = client.options(headers=ELASTIC_JSON_HEADERS)
    prompt = _build_grounded_prompt(claim, hits)
    if settings.llm_task_type == "chat_completion":
        response = request_client.perform_request(
            "POST",
            f"/_inference/chat_completion/{settings.llm_inference_id}",
            body={
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": settings.llm_max_tokens,
                "temperature": 0,
            },
        )
        body = response.body if hasattr(response, "body") else response
        raw_answer = _parse_chat_completion_response(body)
    else:
        response = request_client.perform_request(
            "POST",
            f"/_inference/completion/{settings.llm_inference_id}",
            body={
                "input": prompt,
                "task_settings": {
                    "max_tokens": settings.llm_max_tokens,
                    "temperature": 0,
                },
            },
        )
        body = response.body if hasattr(response, "body") else response
        raw_answer = _parse_completion_response(body)
    verdict, explanation = _parse_agent_json(raw_answer)
    return verdict, explanation, raw_answer
