from fastapi import FastAPI
from pydantic import BaseModel

from src.claim_check import check_claim
from src.live_web_check import live_check_claim
from src.models import ClaimCheckResult
from src.orchestration import VerificationGraph
from src.retrieval import ElasticsearchRetrievalProvider
from src.schemas import SourceEnvelope, SourceRetrievalRequest, VerificationRequest, VerificationResponse
from src.utils import configure_logging


configure_logging()

app = FastAPI(
    title="Bengaluru Misinformation Verification API",
    description=(
        "Agentic, evidence-grounded claim verification with prompt-injection isolation, "
        "citation validation, confidence scoring, and bounded re-query."
    ),
    version="2.0.0",
)


class ClaimRequest(BaseModel):
    claim: str
    retrieval_size: int = 30
    top_n: int = 5


@app.post("/check", response_model=ClaimCheckResult)
def check(request: ClaimRequest) -> ClaimCheckResult:
    return check_claim(
        request.claim,
        retrieval_size=request.retrieval_size,
        top_n=request.top_n,
    )


@app.post("/live-check", response_model=ClaimCheckResult)
def live_check(request: ClaimRequest) -> ClaimCheckResult:
    return live_check_claim(
        request.claim,
        search_results=request.retrieval_size,
        top_n=request.top_n,
    )


@app.post("/provide_sources", response_model=list[SourceEnvelope])
def provide_sources(request: SourceRetrievalRequest) -> list[SourceEnvelope]:
    retrieval_provider = ElasticsearchRetrievalProvider()
    return retrieval_provider.provide_sources(
        request.claim,
        retrieval_size=request.retrieval_size,
        top_k=request.top_k,
    )


@app.post("/verify", response_model=VerificationResponse)
def verify(request: VerificationRequest) -> VerificationResponse:
    verification_graph = VerificationGraph()
    return verification_graph.run(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
