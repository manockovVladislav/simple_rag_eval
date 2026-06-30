from __future__ import annotations

from typing import Any

from rag_eval.config import ModelConfig


def make_huggingface_embeddings(model_config: ModelConfig):
    model_kwargs: dict[str, Any] = {}
    if model_config.local_files_only:
        model_kwargs["local_files_only"] = True

    try:
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

        embeddings = RagasHuggingFaceEmbeddings(
            model=model_config.model,
            device=model_config.embedding_device,
            normalize_embeddings=model_config.normalize_embeddings,
            **model_kwargs,
        )
        return SyncEmbeddingsAdapter(embeddings)
    except ImportError:
        pass

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    if model_config.embedding_device:
        model_kwargs["device"] = model_config.embedding_device

    embeddings = HuggingFaceEmbeddings(
        model_name=model_config.model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": model_config.normalize_embeddings},
        show_progress=False,
    )
    return SyncEmbeddingsAdapter(embeddings)


class SyncEmbeddingsAdapter:
    def __init__(self, embeddings: Any):
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        if hasattr(self.embeddings, "embed_text"):
            return self.embeddings.embed_text(text)
        return self.embeddings.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embeddings, "embed_texts"):
            return self.embeddings.embed_texts(texts)
        return self.embeddings.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
