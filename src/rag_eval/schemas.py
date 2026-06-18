from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextItem:
    text: str
    id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "ContextItem":
        if isinstance(value, ContextItem):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            item_id = value.get("id") or value.get("doc_id") or value.get("document_id")
            text = value.get("text") or value.get("content") or value.get("page_content") or ""
            score = value.get("score")
            metadata = value.get("metadata") or {}
            return cls(text=str(text), id=str(item_id) if item_id is not None else None, score=score, metadata=metadata)
        text = getattr(value, "page_content", None) or getattr(value, "text", None) or str(value)
        item_id = getattr(value, "id", None) or getattr(value, "doc_id", None)
        metadata = getattr(value, "metadata", {}) or {}
        score = getattr(value, "score", None)
        return cls(text=str(text), id=str(item_id) if item_id is not None else None, score=score, metadata=metadata)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PipelineResult:
    answer: str | None = None
    retriever_contexts: list[ContextItem] = field(default_factory=list)
    reranker_contexts: list[ContextItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "PipelineResult":
        if isinstance(value, PipelineResult):
            return value
        if not isinstance(value, dict):
            raise TypeError("Pipeline result must be PipelineResult or dict.")

        answer = value.get("answer") or value.get("model_answer") or value.get("response")
        retriever_contexts = value.get("retriever_contexts") or value.get("retrieved_contexts") or value.get("retriever") or []
        reranker_contexts = value.get("reranker_contexts") or value.get("reranked_contexts") or value.get("reranker") or []
        metadata = value.get("metadata") or {}

        return cls(
            answer=str(answer) if answer is not None else None,
            retriever_contexts=[ContextItem.from_any(item) for item in retriever_contexts],
            reranker_contexts=[ContextItem.from_any(item) for item in reranker_contexts],
            metadata=metadata,
        )


class RagPipeline:
    def answer_question(self, question: str) -> PipelineResult | dict[str, Any]:
        raise NotImplementedError
