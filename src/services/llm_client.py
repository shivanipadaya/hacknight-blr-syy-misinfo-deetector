import json
import logging
from typing import Any

from src.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI/Claude-compatible chat facade backed by Elastic Inference."""

    def __init__(self, client: Any | None = None) -> None:
        from src.elastic_client import get_elasticsearch_client

        self.settings = get_settings()
        self.client = client or get_elasticsearch_client()

    def chat_json(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]:
        raw = self.chat(messages, max_tokens=max_tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("llm_json_parse_failed", extra={"raw_preview": raw[:400]})
            return {}

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
        from src.elastic_client import ELASTIC_JSON_HEADERS

        request_client = self.client.options(headers=ELASTIC_JSON_HEADERS)
        max_tokens = max_tokens or self.settings.llm_max_tokens
        if self.settings.llm_task_type == "chat_completion":
            response = request_client.perform_request(
                "POST",
                f"/_inference/chat_completion/{self.settings.llm_inference_id}",
                body={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
            )
            body = response.body if hasattr(response, "body") else response
            return self._parse_chat_completion_response(body)

        response = request_client.perform_request(
            "POST",
            f"/_inference/completion/{self.settings.llm_inference_id}",
            body={
                "input": "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages),
                "task_settings": {
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
            },
        )
        body = response.body if hasattr(response, "body") else response
        completion = body.get("completion") or []
        if completion:
            return completion[0].get("result", "")
        return body.get("content") or body.get("result") or ""

    @staticmethod
    def _parse_chat_completion_response(body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return body.get("content") or body.get("result") or ""
