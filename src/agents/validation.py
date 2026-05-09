import logging

from src.schemas.verification import FinalVerdict, SourceEnvelope
from src.validators import validate_final_verdict

logger = logging.getLogger(__name__)


class ValidationAgent:
    def run(self, verdict: FinalVerdict, sources: list[SourceEnvelope]) -> FinalVerdict:
        logger.info(
            "validation_agent_started",
            extra={
                "input_verdict": verdict.verdict.value,
                "input_confidence": verdict.confidence,
                "citation_count": len(verdict.citations),
                "source_count": len(sources),
            },
        )
        validated = validate_final_verdict(verdict, sources)
        logger.info(
            "validation_agent_completed",
            extra={
                "output_verdict": validated.verdict.value,
                "output_confidence": validated.confidence,
                "citation_count": len(validated.citations),
            },
        )
        return validated
