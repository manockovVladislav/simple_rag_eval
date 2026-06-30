import faiss
import pickle
import numpy as np
import torch

from typing import List, Dict, Any, Optional, Literal

from retrievers.local_hash_embeddings import HashSentenceTransformer

try:
    import bm25s
except ImportError:
    from retrievers import local_bm25s as bm25s


class HybridRetriever_all_formulas:
    def __init__(
        self,
        faiss_path: str,
        pkl_path: str,
        model_name: str = "bge-m3",
        k_rrf: int = 60,
        fusion_method: Literal[
            "rrf",
            "minmax",
            "zscore",
            "borda",
            "max",
            "min",
            "product",
            "linear_bias",
        ] = "rrf",
        device: Optional[str] = None,
        alpha: float = 0.5,
        bias: float = 0.0,
    ):
        """
        Инициализация гибридного ретривера.

        Args:
            faiss_path: Путь к индексу FAISS (.faiss или .index).
            pkl_path: Путь к метаданным (.pkl).
            model_name: 'bge-m3', 'e5-large' или 'local-hash'.
            k_rrf: Константа сглаживания для RRF.
            fusion_method: Метод слияния.
            device: 'cuda' или 'cpu'. Если None — авто.
            alpha: Вес плотного поиска. 0.5 = баланс.
            bias: Смещение для linear_bias метода.
        """
        self.k_rrf = k_rrf
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.bias = bias

        # Автоопределение устройства
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"Инициализация: {self.device.upper()} | "
            f"Fusion: {self.fusion_method.upper()} | "
            f"α={alpha}"
        )

        self._load_model(model_name)
        self._load_index_and_corpus(faiss_path, pkl_path)
        self._init_bm25()
        self._move_to_device()

    def _load_model(self, model_name: str):
        """Загрузка модели эмбеддингов через sentence-transformers"""

        if model_name == "local-hash":
            self.model = HashSentenceTransformer()
            self.query_prefix = ""

        elif model_name == "bge-m3":
            from sentence_transformers import SentenceTransformer

            # trust_remote_code=True обязателен для корректной работы bge-m3
            self.model = SentenceTransformer(
                "models/bge-m3/BAAI_bge-m3",
                trust_remote_code=True,
            )
            self.query_prefix = ""

        elif model_name == "e5-large":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("models/intfloat/e5-large-v2")
            self.query_prefix = "query: "

        else:
            raise ValueError("Поддерживаются только: 'bge-m3', 'e5-large', 'local-hash'")

    def _load_index_and_corpus(self, faiss_path: str, pkl_path: str):
        """Загрузка индекса и корпуса"""

        self.index = faiss.read_index(faiss_path)

        with open(pkl_path, "rb") as f:
            self.corpus = pickle.load(f)

        # Универсальное извлечение текста из структуры PKL
        self.texts = [
            doc if isinstance(doc, str)
            else doc.get("text", doc.get("content", str(doc)))
            for doc in self.corpus
        ]

        assert self.index.ntotal == len(self.texts), (
            "❌ Размер FAISS и PKL не совпадает!"
        )

        print(f"✅ Загружено {self.index.ntotal} документов")

    def _init_bm25(self):
        """Индексация BM25, всегда работает на CPU"""

        self.bm25 = bm25s.BM25()

        # bm25s.tokenize использует быструю токенизацию
        self.bm25.index(bm25s.tokenize(self.texts))

    def _move_to_device(self):
        """Безопасный перенос модели и FAISS на GPU"""

        self.model.to(self.device)

        if self.device == "cuda":
            try:
                if not hasattr(faiss, "StandardGpuResources"):
                    raise AttributeError(
                        "faiss-gpu не найден. Установите faiss-gpu-cu12."
                    )

                self.gpu_res = faiss.StandardGpuResources()

                # index_cpu_to_gpu копирует индекс в VRAM для ускорения
                self.index = faiss.index_cpu_to_gpu(
                    self.gpu_res,
                    0,
                    self.index,
                )

                print("✅ FAISS индекс успешно перенесен на GPU 0")

            except Exception as e:
                print(f"⚠️ FAISS GPU недоступен ({e}). Переключаюсь на CPU.")
                self.device = "cpu"
                self.model.to("cpu")

    def _normalize_scores(
        self,
        d_scores: np.ndarray,
        s_scores: np.ndarray,
        method: str = "minmax",
    ):
        """Вспомогательный метод нормализации скоров"""

        if method == "minmax":
            d_norm = (
                (d_scores - d_scores.min())
                / (d_scores.max() - d_scores.min() + 1e-8)
            )
            s_norm = (
                (s_scores - s_scores.min())
                / (s_scores.max() - s_scores.min() + 1e-8)
            )

        elif method == "zscore":
            d_norm = (
                (d_scores - d_scores.mean())
                / (d_scores.std() + 1e-8)
            )
            s_norm = (
                (s_scores - s_scores.mean())
                / (s_scores.std() + 1e-8)
            )

        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")

        return d_norm, s_norm

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Выполняет гибридный поиск.

        Args:
            query: Текст запроса.
            top_k: Количество возвращаемых документов.
            alpha: Переопределение веса alpha для конкретного запроса.

        Returns:
            Список найденных документов.
        """

        effective_alpha = alpha if alpha is not None else self.alpha

        # 1. Dense поиск через FAISS
        q_text = f"{self.query_prefix}{query}"

        q_vec = self.model.encode(
            [q_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        q_vec = np.asarray(q_vec, dtype="float32")

        # D = дистанции / скоры, I = индексы
        D_dense, I_dense = self.index.search(q_vec, top_k)

        # 2. Lexical поиск через BM25
        q_tokens = bm25s.tokenize(query)

        I_sparse, D_sparse = self.bm25.retrieve(
            q_tokens,
            k=top_k,
        )

        # Приведение к одномерным массивам numpy
        d_scores = D_dense[0].astype(float)
        s_scores = D_sparse[0].astype(float)

        d_ids = I_dense[0]
        s_ids = I_sparse[0]

        # Словари рангов для методов, работающих с рангами
        dense_ranks = {
            int(idx): rank
            for rank, idx in enumerate(d_ids)
        }

        sparse_ranks = {
            int(idx): rank
            for rank, idx in enumerate(s_ids)
        }

        # Объединяем все уникальные ID из обоих поисков
        all_ids = set(map(int, d_ids)) | set(map(int, s_ids))

        scores = {}
        method = self.fusion_method

        # --- 1. RRF / Reciprocal Rank Fusion ---
        if method == "rrf":
            for idx in all_ids:
                r_d = dense_ranks.get(idx, top_k)
                r_s = sparse_ranks.get(idx, top_k)

                scores[idx] = (
                    effective_alpha / (r_d + self.k_rrf)
                    + (1 - effective_alpha) / (r_s + self.k_rrf)
                )

        # --- 2. Min-Max + Linear ---
        elif method == "minmax":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "minmax",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            for idx in all_ids:
                scores[idx] = (
                    effective_alpha * d_map.get(idx, 0.0)
                    + (1 - effective_alpha) * s_map.get(idx, 0.0)
                )

        # --- 3. Z-Score + Linear ---
        elif method == "zscore":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "zscore",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            # Штраф для документов, отсутствующих в одном из списков
            d_pen = float(d_norm.min() - 1.0)
            s_pen = float(s_norm.min() - 1.0)

            for idx in all_ids:
                scores[idx] = (
                    effective_alpha * d_map.get(idx, d_pen)
                    + (1 - effective_alpha) * s_map.get(idx, s_pen)
                )

        # --- 4. Borda Count ---
        elif method == "borda":
            N = top_k

            for idx in all_ids:
                r_d = dense_ranks.get(idx, N)
                r_s = sparse_ranks.get(idx, N)

                scores[idx] = (
                    effective_alpha * (N - 1 - r_d)
                    + (1 - effective_alpha) * (N - 1 - r_s)
                )

        # --- 5. MAX / OR-логика ---
        elif method == "max":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "minmax",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            for idx in all_ids:
                scores[idx] = max(
                    d_map.get(idx, 0.0),
                    s_map.get(idx, 0.0),
                )

        # --- 6. MIN / AND-логика ---
        elif method == "min":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "minmax",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            for idx in all_ids:
                scores[idx] = min(
                    d_map.get(idx, 0.0),
                    s_map.get(idx, 0.0),
                )

        # --- 7. PRODUCT / произведение ---
        elif method == "product":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "minmax",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            for idx in all_ids:
                scores[idx] = (
                    d_map.get(idx, 0.0)
                    * s_map.get(idx, 0.0)
                )

        # --- 8. LINEAR + BIAS ---
        elif method == "linear_bias":
            d_norm, s_norm = self._normalize_scores(
                d_scores,
                s_scores,
                "minmax",
            )

            d_map = dict(zip(map(int, d_ids), d_norm))
            s_map = dict(zip(map(int, s_ids), s_norm))

            for idx in all_ids:
                scores[idx] = (
                    effective_alpha * d_map.get(idx, 0.0)
                    + (1 - effective_alpha) * s_map.get(idx, 0.0)
                    + self.bias
                )

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        # 4. Сортировка и формирование результата
        sorted_ids = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []

        for idx, score in sorted_ids:
            doc = self.corpus[idx]

            if isinstance(doc, str):
                text = doc
                source = ""
                file_path = ""
                chunk_id = idx
            else:
                text = doc.get("text", "")
                source = doc.get("source", "")
                file_path = doc.get("file_path", "")
                chunk_id = doc.get("chunk_id", idx)

            results.append({
                "score": round(float(score), 4),
                "text": text,
                "source": source,
                "file_path": file_path,
                "chunk_id": chunk_id,
            })

        return results

    def set_bias(self, b: float):
        """Изменить bias для метода linear_bias"""

        self.bias = b
        print(f"✅ bias изменён на: {b}")
