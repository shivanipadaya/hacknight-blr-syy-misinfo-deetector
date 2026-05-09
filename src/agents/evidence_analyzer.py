import logging

from pydantic import ValidationError

from src.prompts import build_evidence_analysis_messages
from src.schemas.verification import EvidenceAssessment, EvidenceLabel, SourceEnvelope
from src.services import LLMClient

logger = logging.getLogger(__name__)


class EvidenceAnalyzerAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, claim: str, sources: list[SourceEnvelope]) -> list[EvidenceAssessment]:
        if not sources:
            logger.info("evidence_analyzer_skipped", extra={"reason": "no_sources"})
            return []

        logger.info(
            "evidence_analyzer_started",
            extra={"source_count": len(sources), "source_ids": [source.id for source in sources]},
        )
        payload = self.llm.chat_json(build_evidence_analysis_messages(claim, sources))
        raw_assessments = payload.get("assessments") or []
        assessments: list[EvidenceAssessment] = []
        valid_source_ids = {source.id for source in sources}
        for item in raw_assessments:
            try:
                assessment = EvidenceAssessment.model_validate(item)
            except ValidationError as exc:
                logger.warning("assessment_validation_failed", extra={"error": str(exc)})
                continue
            if assessment.source_id in valid_source_ids:
                assessments.append(assessment)

        assessed_ids = {item.source_id for item in assessments}
        for source in sources:
            if source.id not in assessed_ids:
                assessments.append(
                    EvidenceAssessment(
                        source_id=source.id,
                        label=EvidenceLabel.not_mentioned,
                        rationale="The analyzer did not return a valid assessment for this source.",
                        relevance=0.0,
                    )
                )
        logger.info(
            "evidence_analyzer_completed",
            extra={
                "assessment_count": len(assessments),
                "labels": [assessment.label.value for assessment in assessments],
            },
        )
        return assessments
