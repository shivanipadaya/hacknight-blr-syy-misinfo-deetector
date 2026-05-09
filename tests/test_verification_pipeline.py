from src.agents.confidence import ConfidenceAgent
from src.schemas.verification import (
    CitationRef,
    EvidenceAssessment,
    EvidenceLabel,
    FinalVerdict,
    SourceEnvelope,
    VerdictLabel,
)
from src.validators import validate_final_verdict


def test_invalid_citations_downgrade_to_unverified() -> None:
    verdict = FinalVerdict(
        verdict=VerdictLabel.supported,
        confidence=0.82,
        summary="Supported by retrieved evidence.",
        citations=[CitationRef(id="missing", url="https://example.com", date_of_scraping="2026-05-09")],
    )
    sources = [
        SourceEnvelope(
            id="doc_1",
            url="https://example.com/real",
            text="BMRCL service update.",
            date_of_scraping="2026-05-09",
        )
    ]

    validated = validate_final_verdict(verdict, sources)

    assert validated.verdict == VerdictLabel.unverified
    assert validated.confidence < 0.5
    assert validated.citations == []


def test_valid_citations_are_normalized_from_sources() -> None:
    verdict = FinalVerdict(
        verdict=VerdictLabel.supported,
        confidence=0.82,
        summary="Supported by retrieved evidence.",
        citations=[CitationRef(id="doc_1", url="https://wrong.example", date_of_scraping=None)],
    )
    sources = [
        SourceEnvelope(
            id="doc_1",
            url="https://example.com/real",
            text="BMRCL service update.",
            date_of_scraping="2026-05-09",
        )
    ]

    validated = validate_final_verdict(verdict, sources)

    assert validated.verdict == VerdictLabel.supported
    assert validated.citations[0].url == "https://example.com/real"
    assert validated.citations[0].date_of_scraping == "2026-05-09"


def test_confidence_penalizes_no_meaningful_evidence() -> None:
    report = ConfidenceAgent().run(
        [
            EvidenceAssessment(source_id="doc_1", label=EvidenceLabel.not_mentioned, relevance=0.0),
            EvidenceAssessment(source_id="doc_2", label=EvidenceLabel.not_mentioned, relevance=0.0),
        ],
        [
            SourceEnvelope(id="doc_1", url="https://example.com/1", text="No mention."),
            SourceEnvelope(id="doc_2", url="https://example.com/2", text="No mention."),
        ],
    )

    assert report.confidence <= 0.25
