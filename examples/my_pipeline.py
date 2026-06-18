from __future__ import annotations

from rag_eval import ContextItem, PipelineResult


class ExamplePipeline:
    def answer_question(self, question: str) -> PipelineResult:
        retriever_contexts = [
            ContextItem(id="doc_1", text="Пример найденного документа.", score=0.9),
            ContextItem(id="doc_2", text="Еще один контекст ретривера.", score=0.8),
        ]
        reranker_contexts = [
            ContextItem(id="doc_1", text="Пример найденного документа.", score=0.95),
        ]
        return PipelineResult(
            answer=f"Демо-ответ на вопрос: {question}",
            retriever_contexts=retriever_contexts,
            reranker_contexts=reranker_contexts,
            metadata={"pipeline": "example"},
        )


def create_pipeline(config=None) -> ExamplePipeline:
    return ExamplePipeline()
