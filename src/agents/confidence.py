from collections import Counter
import logging

from src.schemas.verification import ConfidenceReport, EvidenceAssessment, EvidenceLabel, SourceEnvelope, VerdictLabel

logger = logging.getLogger(__name__)


class ConfidenceAgent:
    def run(self, assessments: list[EvidenceAssessment], sources: list[SourceEnvelope]) -> ConfidenceReport:
        if not assessments or not sources:
            logger.info(
                "confidence_agent_completed",
                extra={"confidence": 0.0, "reason": "missing_assessments_or_sources"},
            )
            return ConfidenceReport(confidence=0.0, rationale="No sources were available for assessment.")

        counts = Counter(item.label for item in assessments)
        meaningful = len(assessments) - counts[EvidenceLabel.not_mentioned]
        avg_relevance = sum(item.relevance for item in assessments) / max(len(assessments), 1)
        agreement = max(
            counts[EvidenceLabel.supports],
            counts[EvidenceLabel.contradicts],
            counts[EvidenceLabel.partially_supports],
        ) / max(meaningful, 1)
        coverage = meaningful / max(len(sources), 1)
        confidence = (0.45 * agreement) + (0.35 * avg_relevance) + (0.20 * coverage)
        if meaningful == 0:
            confidence = min(confidence, 0.25)
        if counts[EvidenceLabel.supports] and counts[EvidenceLabel.contradicts]:
            confidence = min(confidence, 0.68)

        report = ConfidenceReport(
            confidence=round(max(0.0, min(confidence, 1.0)), 2),
            rationale=(
                f"coverage={coverage:.2f}, agreement={agreement:.2f}, "
                f"average_relevance={avg_relevance:.2f}"
            ),
        )
        logger.info(
            "confidence_agent_completed",
            extra={
                "confidence": report.confidence,
                "meaningful_assessments": meaningful,
                "source_count": len(sources),
                "rationale": report.rationale,
            },
        )
        return report

    @staticmethod
    def inferred_verdict(assessments: list[EvidenceAssessment]) -> VerdictLabel:
        counts = Counter(item.label for item in assessments)
        if counts[EvidenceLabel.supports] and not counts[EvidenceLabel.contradicts]:
            return VerdictLabel.supported
        if counts[EvidenceLabel.contradicts] and not counts[EvidenceLabel.supports]:
            return VerdictLabel.unsupported
        if counts[EvidenceLabel.partially_supports] or (
            counts[EvidenceLabel.supports] and counts[EvidenceLabel.contradicts]
        ):
            return VerdictLabel.partially_supported
        return VerdictLabel.unverified
