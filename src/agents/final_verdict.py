import logging

from pydantic import ValidationError

from src.agents.confidence import ConfidenceAgent
from src.prompts import build_final_verdict_messages
from src.schemas.verification import CitationRef, EvidenceAssessment, FinalVerdict, SourceEnvelope
from src.services import LLMClient

logger = logging.getLogger(__name__)


class FinalVerdictAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(
        self,
        claim: str,
        sources: list[SourceEnvelope],
        assessments: list[EvidenceAssessment],
        confidence: float,
    ) -> FinalVerdict:
        logger.info(
            "final_verdict_agent_started",
            extra={
                "source_count": len(sources),
                "assessment_count": len(assessments),
                "confidence": confidence,
            },
        )
        payload = self.llm.chat_json(build_final_verdict_messages(claim, sources, assessments, confidence))
        try:
            verdict = FinalVerdict.model_validate(payload)
            logger.info(
                "final_verdict_agent_completed",
                extra={
                    "verdict": verdict.verdict.value,
                    "confidence": verdict.confidence,
                    "citation_count": len(verdict.citations),
                    "fallback": False,
                },
            )
            return verdict
        except ValidationError as exc:
            logger.warning("final_verdict_validation_failed", extra={"error": str(exc)})
            inferred = ConfidenceAgent.inferred_verdict(assessments)
            citations = self._fallback_citations(sources, assessments)
            verdict = FinalVerdict(
                verdict=inferred,
                confidence=confidence,
                summary="The verdict was generated from validated per-source assessments because the LLM final response was malformed.",
                citations=citations,
            )
            logger.info(
                "final_verdict_agent_completed",
                extra={
                    "verdict": verdict.verdict.value,
                    "confidence": verdict.confidence,
                    "citation_count": len(verdict.citations),
                    "fallback": True,
                },
            )
            return verdict

    @staticmethod
    def _fallback_citations(
        sources: list[SourceEnvelope],
        assessments: list[EvidenceAssessment],
    ) -> list[CitationRef]:
        source_by_id = {source.id: source for source in sources}
        ranked = sorted(assessments, key=lambda item: item.relevance, reverse=True)
        citations: list[CitationRef] = []
        for assessment in ranked:
            if assessment.relevance <= 0:
                continue
            source = source_by_id.get(assessment.source_id)
            if source is None:
                continue
            citations.append(
                CitationRef(
                    id=source.id,
                    url=source.url,
                    date_of_scraping=source.date_of_scraping,
                )
            )
            if len(citations) == 3:
                break
        return citations
