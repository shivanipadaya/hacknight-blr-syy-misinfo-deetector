from src.schemas.verification import CitationRef, FinalVerdict, SourceEnvelope, VerdictLabel


def validate_final_verdict(verdict: FinalVerdict, sources: list[SourceEnvelope]) -> FinalVerdict:
    valid_by_id = {source.id: source for source in sources}
    valid_citations: list[CitationRef] = []
    for citation in verdict.citations:
        source = valid_by_id.get(citation.id)
        if source is None:
            continue
        valid_citations.append(
            CitationRef(
                id=source.id,
                url=source.url,
                date_of_scraping=source.date_of_scraping,
            )
        )

    if not valid_citations:
        return FinalVerdict(
            verdict=VerdictLabel.unverified,
            confidence=min(verdict.confidence, 0.49),
            summary="The generated verdict did not include valid source citations, so it was downgraded to UNVERIFIED.",
            citations=[],
        )

    return verdict.model_copy(update={"citations": valid_citations})
