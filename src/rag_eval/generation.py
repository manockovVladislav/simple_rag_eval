from __future__ import annotations

from typing import Any

from rag_eval.config import AppConfig
from rag_eval.llm_clients import ChatMessage, make_model_client
from rag_eval.schemas import ContextItem, PipelineResult


class RagAnswerGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Any | None = None

    def generate(self, question: str, result: PipelineResult) -> str | None:
        contexts = self.contexts_for_generation(result)
        if not contexts:
            return None

        messages = [
            ChatMessage(role="system", content=self.config.generation.system_prompt),
            ChatMessage(role="user", content=self._user_prompt(question, contexts)),
        ]
        return self._model().chat(messages).strip()

    def contexts_for_generation(self, result: PipelineResult) -> list[ContextItem]:
        source = self.config.generation.context_source
        if source == "retriever":
            primary = result.retriever_contexts
            fallback = result.reranker_contexts
        else:
            primary = result.reranker_contexts
            fallback = result.retriever_contexts
        contexts = primary or fallback
        return contexts[: self.config.generation.max_contexts]

    def _model(self):
        if self._client is None:
            self._client = make_model_client(self.config.generation.provider, self.config.models)
        return self._client

    def _user_prompt(self, question: str, contexts: list[ContextItem]) -> str:
        context_text = "\n\n".join(
            f"[{index}] {context.text}" for index, context in enumerate(contexts, start=1)
        )
        return (
            "Контекст:\n"
            f"{context_text}\n\n"
            "Вопрос:\n"
            f"{question}\n\n"
            "Ответ:"
        )
