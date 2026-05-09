import logging
import re

from src.prompts import build_query_reformulation_messages
from src.services import LLMClient

logger = logging.getLogger(__name__)


class QueryReformulationAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, claim: str, prior_queries: list[str]) -> str:
        logger.info(
            "query_reformulation_started",
            extra={"prior_query_count": len(prior_queries), "claim_preview": claim[:160]},
        )
        payload = self.llm.chat_json(build_query_reformulation_messages(claim, prior_queries), max_tokens=120)
        query = str(payload.get("query") or "").strip()
        if query:
            refined = query[:240]
            logger.info("query_reformulation_completed", extra={"query": refined, "fallback": False})
            return refined
        refined = self._fallback_query(claim)
        logger.info("query_reformulation_completed", extra={"query": refined, "fallback": True})
        return refined

    @staticmethod
    def _fallback_query(claim: str) -> str:
        query = re.sub(r"[\[\]{}<>\"']", " ", claim)
        query = re.sub(r"\b(ignore|instructions|system|developer|prompt)\b", " ", query, flags=re.I)
        query = " ".join(query.split())
        return query[:240] or claim[:240]
