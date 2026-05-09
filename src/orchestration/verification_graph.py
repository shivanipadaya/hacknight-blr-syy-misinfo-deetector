import logging

from src.agents.confidence import ConfidenceAgent
from src.agents.evidence_analyzer import EvidenceAnalyzerAgent
from src.agents.final_verdict import FinalVerdictAgent
from src.agents.query_reformulation import QueryReformulationAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.validation import ValidationAgent
from src.injection_guard import detect_prompt_injection
from src.schemas.verification import VerificationRequest, VerificationResponse, VerdictLabel

logger = logging.getLogger(__name__)


class VerificationGraph:
    """Deterministic agent orchestration graph for claim verification."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent | None = None,
        evidence_agent: EvidenceAnalyzerAgent | None = None,
        confidence_agent: ConfidenceAgent | None = None,
        reformulation_agent: QueryReformulationAgent | None = None,
        validation_agent: ValidationAgent | None = None,
        final_agent: FinalVerdictAgent | None = None,
    ) -> None:
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.evidence_agent = evidence_agent or EvidenceAnalyzerAgent()
        self.confidence_agent = confidence_agent or ConfidenceAgent()
        self.reformulation_agent = reformulation_agent or QueryReformulationAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        self.final_agent = final_agent or FinalVerdictAgent()

    def run(self, request: VerificationRequest) -> VerificationResponse:
        logger.info(
            "verification_graph_started",
            extra={
                "claim_preview": request.claim[:160],
                "retrieval_size": request.retrieval_size,
                "top_k": request.top_k,
                "max_requery_attempts": request.max_requery_attempts,
            },
        )
        query = request.claim
        attempted_queries: list[str] = []
        refined_queries: list[str] = []
        warnings = detect_prompt_injection(request.claim)
        best_response: VerificationResponse | None = None

        for attempt in range(request.max_requery_attempts + 1):
            attempted_queries.append(query)
            logger.info("verification_attempt", extra={"attempt": attempt, "query": query})
            sources = self.retrieval_agent.run(
                query,
                retrieval_size=request.retrieval_size,
                top_k=request.top_k,
            )
            for source in sources:
                warnings.extend(detect_prompt_injection(source.text))

            assessments = self.evidence_agent.run(request.claim, sources)
            confidence_report = self.confidence_agent.run(assessments, sources)
            final = self.final_agent.run(request.claim, sources, assessments, confidence_report.confidence)
            final = self.validation_agent.run(final, sources)
            response = VerificationResponse(
                claim=request.claim,
                verdict=final.verdict,
                confidence=final.confidence,
                summary=final.summary,
                citations=final.citations,
                attempts=attempt,
                analyzed_sources=assessments,
                refined_queries=refined_queries,
                injection_warnings=sorted(set(warnings)),
            )
            if best_response is None or response.confidence > best_response.confidence:
                best_response = response

            if response.confidence >= 0.5 or attempt >= request.max_requery_attempts:
                logger.info(
                    "verification_graph_completed",
                    extra={
                        "attempts": attempt,
                        "verdict": response.verdict.value,
                        "confidence": response.confidence,
                        "citation_count": len(response.citations),
                    },
                )
                return response

            query = self.reformulation_agent.run(request.claim, attempted_queries)
            if query in attempted_queries:
                response = best_response.model_copy(
                    update={
                        "verdict": VerdictLabel.unverified,
                        "confidence": min(best_response.confidence, 0.49),
                        "summary": "The system could not produce a distinct refined query with enough confidence.",
                    }
                )
                logger.info(
                    "verification_graph_completed",
                    extra={
                        "attempts": attempt,
                        "verdict": response.verdict.value,
                        "confidence": response.confidence,
                        "reason": "duplicate_refined_query",
                    },
                )
                return response
            refined_queries.append(query)

        response = best_response or VerificationResponse(
            claim=request.claim,
            verdict=VerdictLabel.unverified,
            confidence=0.0,
            summary="The verification workflow did not complete.",
            citations=[],
        )
        logger.info(
            "verification_graph_completed",
            extra={
                "attempts": response.attempts,
                "verdict": response.verdict.value,
                "confidence": response.confidence,
                "reason": "exhausted",
            },
        )
        return response
