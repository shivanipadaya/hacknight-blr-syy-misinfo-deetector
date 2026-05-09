from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class Verdict(StrEnum):
    supported = "SUPPORTED"
    contradicted = "CONTRADICTED"
    partially_supported = "PARTIALLY_SUPPORTED"
    insufficient_evidence = "INSUFFICIENT_EVIDENCE"


class SourceDocument(BaseModel):
    url: HttpUrl | str
    source_name: str = "unknown"
    source_type: str = "unknown"
    title: str = ""
    content: str
    published_at: datetime | None = None
    crawled_at: datetime | None = None
    language: str = "en"
    content_hash: str | None = None


class SearchHit(BaseModel):
    doc_id: str
    url: str
    title: str = ""
    source_name: str = "unknown"
    source_type: str = "unknown"
    published_at: str | None = None
    snippet: str
    score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    title: str
    url: str
    source_name: str
    published_at: str | None = None
    quoted_snippet: str
    relevance_score: float = 0.0


class RelevantResource(BaseModel):
    resource: str
    data: dict
    metadata: dict


class ClaimCheckResult(BaseModel):
    claim: str
    verdict: Verdict
    explanation: str
    agent_answer: str | None = None
    resources: list[RelevantResource] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    injection_warnings: list[str] = Field(default_factory=list)
