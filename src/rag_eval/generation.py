from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag_eval.config import AppConfig
from rag_eval.llm_clients import ChatMessage, make_model_client
from rag_eval.schemas import ContextItem, PipelineResult


class RagAnswerGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._clients: dict[str, Any] = {}

    def generate(self, question: str, result: PipelineResult) -> str | None:
        return self.generate_for_provider(self.config.generation.provider, question, result)

    def generate_for_provider(self, provider: str, question: str, result: PipelineResult) -> str | None:
        contexts = self.contexts_for_generation(result)
        if not contexts:
            return None

        messages = [
            ChatMessage(role="system", content=self.config.generation.system_prompt),
            ChatMessage(role="user", content=self._user_prompt(question, contexts)),
        ]
        return self._model(provider).chat(messages).strip()

    def generate_many(self, question: str, result: PipelineResult) -> dict[str, dict[str, str | None]]:
        """Issue independent blocking API calls concurrently in a bounded thread pool."""
        providers = list(dict.fromkeys(
            self.config.generation.providers or [self.config.generation.provider]
        ))
        max_workers = min(len(providers), max(1, self.config.generation.max_workers))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-llm") as executor:
            futures = {
                provider: executor.submit(self._safe_generate, provider, question, result)
                for provider in providers
            }
            return {provider: future.result() for provider, future in futures.items()}

    def _safe_generate(
        self,
        provider: str,
        question: str,
        result: PipelineResult,
    ) -> dict[str, str | None]:
        try:
            answer = self.generate_for_provider(provider, question, result)
            return {"answer": answer, "error": None}
        except Exception as exc:
            return {"answer": None, "error": f"{type(exc).__name__}: {exc}"}

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

    def _model(self, provider: str):
        if provider not in self._clients:
            self._clients[provider] = make_model_client(provider, self.config.models)
        return self._clients[provider]

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
