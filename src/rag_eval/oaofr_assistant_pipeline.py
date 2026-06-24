from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rag_eval.config import AppConfig
from rag_eval.schemas import ContextItem, PipelineResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_RETRIEVER_REPO = PROJECT_ROOT / "oaofr_assistant_gr"
OLD_RETRIEVER_REPO = PROJECT_ROOT / "retriver_gr_oaofr"

for repo_path in (NEW_RETRIEVER_REPO, OLD_RETRIEVER_REPO):
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

if NEW_RETRIEVER_REPO.exists():
    from retrivers.hybrid_retriever_new_all_formulas import (  # noqa: E402
        HybridRetriever_all_formulas,
    )
    from retrivers.hybrid_retriever_new_all_formulas_and_reranker import (  # noqa: E402
        HybridRetriever_all_formulas_and_reranker,
    )
else:
    from retriver.hibrid_retriver_all_formulas import (  # type: ignore[no-redef]  # noqa: E402
        HybridRetriever_all_formulas,
    )
    from retriver.hibrid_retriver_all_formulas_and_reranker import (  # type: ignore[no-redef]  # noqa: E402
        HybridRetriever_all_formulas_and_reranker,
    )


class RagEvalHybridRetrieverOnly(HybridRetriever_all_formulas):
    def __init__(
        self,
        *args: Any,
        eval_top_k: int = 10,
        eval_search_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.eval_top_k = eval_top_k
        self.eval_search_kwargs = eval_search_kwargs or {}

    def answer_question(self, question: str) -> PipelineResult:
        retriever_contexts = self._search(question)
        return PipelineResult(
            answer=None,
            retriever_contexts=[ContextItem.from_any(item) for item in retriever_contexts],
            reranker_contexts=[],
            metadata={"pipeline": type(self).__name__},
        )

    def _search(self, question: str) -> list[dict[str, Any]]:
        result = self.search(
            question,
            top_k=self.eval_top_k,
            **self.eval_search_kwargs,
        )
        return [] if result is None else list(result)


class RagEvalHybridRetrieverWithReranker(HybridRetriever_all_formulas_and_reranker):
    def __init__(
        self,
        *args: Any,
        eval_top_k: int = 10,
        eval_search_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.eval_top_k = eval_top_k
        self.eval_search_kwargs = eval_search_kwargs or {}

    def answer_question(self, question: str) -> PipelineResult:
        if self.reranker is None:
            retriever_contexts = self._search(question)
            reranker_contexts = []
        else:
            retriever_contexts = self._search_without_reranker(question)
            reranker_contexts = self._search(question)

        return PipelineResult(
            answer=None,
            retriever_contexts=[ContextItem.from_any(item) for item in retriever_contexts],
            reranker_contexts=[ContextItem.from_any(item) for item in reranker_contexts],
            metadata={"pipeline": type(self).__name__},
        )

    def _search(self, question: str) -> list[dict[str, Any]]:
        result = self.search(
            question,
            top_k=self.eval_top_k,
            **self.eval_search_kwargs,
        )
        return [] if result is None else list(result)

    def _search_without_reranker(self, question: str) -> list[dict[str, Any]]:
        reranker = self.reranker
        self.reranker = None
        try:
            return self._search(question)
        finally:
            self.reranker = reranker


def create_retriever_pipeline(config: AppConfig) -> RagEvalHybridRetrieverOnly:
    adapter = config.retriever_adapter
    return RagEvalHybridRetrieverOnly(
        *adapter.init_args,
        eval_top_k=adapter.top_k,
        eval_search_kwargs=adapter.search_kwargs,
        **adapter.init_kwargs,
    )


def create_reranker_pipeline(config: AppConfig) -> RagEvalHybridRetrieverWithReranker:
    adapter = config.retriever_adapter
    return RagEvalHybridRetrieverWithReranker(
        *adapter.init_args,
        eval_top_k=adapter.top_k,
        eval_search_kwargs=adapter.search_kwargs,
        **adapter.init_kwargs,
    )
