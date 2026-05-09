from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VerdictLabel(StrEnum):
    supported = "SUPPORTED"
    unsupported = "UNSUPPORTED"
    partially_supported = "PARTIALLY_SUPPORTED"
    unverified = "UNVERIFIED"


class EvidenceLabel(StrEnum):
    supports = "SUPPORTS"
    contradicts = "CONTRADICTS"
    partially_supports = "PARTIALLY_SUPPORTS"
    not_mentioned = "NOT_MENTIONED"


class SourceRetrievalRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_size: int = Field(default=30, ge=1, le=100)


class VerificationRequest(SourceRetrievalRequest):
    max_requery_attempts: int = Field(default=2, ge=0, le=2)


class SourceEnvelope(BaseModel):
    id: str = Field(min_length=1)
    url: str = ""
    text: str = Field(default="", max_length=12000)
    date_of_scraping: str | None = None
    title: str = ""
    source_name: str = "unknown"
    source_type: str = "unknown"
    retrieval_score: float = 0.0
    rerank_score: float | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("text")
    @classmethod
    def collapse_text(cls, value: str) -> str:
        return " ".join((value or "").replace("\x00", " ").split())


class CitationRef(BaseModel):
    id: str = Field(min_length=1)
    url: str = ""
    date_of_scraping: str | None = None


class EvidenceAssessment(BaseModel):
    source_id: str = Field(min_length=1)
    label: EvidenceLabel
    rationale: str = Field(default="", max_length=1200)
    cited_quote: str = Field(default="", max_length=800)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class ConfidenceReport(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=1200)


class FinalVerdict(BaseModel):
    verdict: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1600)
    citations: list[CitationRef] = Field(default_factory=list)


class VerificationResponse(FinalVerdict):
    claim: str
    attempts: int = Field(default=0, ge=0)
    analyzed_sources: list[EvidenceAssessment] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)
    injection_warnings: list[str] = Field(default_factory=list)
