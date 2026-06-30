import os
from pathlib import Path


def _load_credentials(path: str = ".credentials") -> None:
    credentials_path = Path(__file__).with_name(path)
    if not credentials_path.exists():
        return
    for line in credentials_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


_load_credentials()

# Пути
GOLDEN_QUESTIONS_PATH = "data/golden_questions.xlsx"
RUN_OUTPUTS_DIR = "outputs/runs"
METRICS_OUTPUTS_DIR = "outputs/metrics"
SUMMARY_METRICS_FILE = "outputs/metrics/metrics_summary.xlsx"
LOG_FILE = "outputs/logs/rag_eval.log"

# Колонки в golden-файле
QUESTION_COLUMN = "question"
GROUND_TRUTH_COLUMN = "ground_truth"
EXPECTED_ANSWER_COLUMN = "expected_answer"  # fallback для старых файлов
CHUNK_ID_COLUMN = "chunk_id"

# Какой pipeline запускать.
#
# 1. Демо-проверка тестовой системы без внешнего ретривера:
# PIPELINE_FACTORY = "examples.my_pipeline:create_pipeline"
#
# 2. Твой файл oaofr_assistant_gr/retrievers/hybrid_retriever_new_all_formulas.py
#    Только retriever, без reranker. Результаты попадут в retriever_contexts.
PIPELINE_FACTORY = "rag_eval.oaofr_assistant_pipeline:create_retriever_pipeline"
#
# 3. Твой файл oaofr_assistant_gr/retrievers/hybrid_retriever_new_all_formulas_and_reranker.py
#    Retriever + reranker. Результаты retriever попадут в retriever_contexts,
#    результаты reranker попадут в reranker_contexts.
# PIPELINE_FACTORY = "rag_eval.oaofr_assistant_pipeline:create_reranker_pipeline"

# Параметры создания твоих классов.
#
# Сюда добавлять ровно те аргументы, которые принимает __init__ класса:
# - faiss_path: путь к локальному FAISS индексу;
# - pkl_path: путь к локальному pkl с корпусом/метаданными;
# - model_name: "bge-m3", "e5-large" или "local-hash" для локальной тестовой базы;
# - fusion_method, alpha, bias, device и другие параметры твоего класса;
# - reranker: True/False только для класса retriever + reranker.
RETRIEVER_INIT_ARGS = []
RETRIEVER_INIT_KWARGS = {
    "faiss_path": "oaofr_assistant_gr/vectors/index.faiss",
    "pkl_path": "oaofr_assistant_gr/vectors/metadata.pkl",
    "model_name": "local-hash",
    "fusion_method": "rrf",
    "device": "cpu",
    # "reranker": True,
}
RETRIEVER_SEARCH_KWARGS = {}
RETRIEVER_TOP_K = 10

# Настройки прогона.
#
# Для быстрой проверки поставь:
# MAX_QUESTIONS = 1
# RAGAS_ENABLED = False
#
# Это проверит, что выбранный класс импортируется, индексы читаются,
# search(...) работает, а результат пишется в outputs/runs/*.xlsx.
#
# После этого включай полный прогон:
# MAX_QUESTIONS = None
# RAGAS_ENABLED = True
MAX_QUESTIONS = None
SLEEP_SECONDS = 0

# Генерация финального RAG-ответа после retriever/reranker.
#
# RAG_LLM_PROVIDER и RAGAS_JUDGE_PROVIDER можно выставлять независимо:
# например, RAG_LLM_PROVIDER = "gigachat", RAGAS_JUDGE_PROVIDER = "qwen".
RAG_ANSWER_ENABLED = True
RAG_LLM_PROVIDER = "qwen"
RAG_CONTEXT_SOURCE = "retriever"
RAG_MAX_CONTEXTS = 10
RAG_SYSTEM_PROMPT = (
    "Ты отвечаешь на вопрос только по переданному контексту. "
    "Если в контексте нет ответа, так и скажи."
)

# Локальные answer-метрики без LLM-судьи
ANSWER_METRICS_ENABLED = True
RETRIEVAL_K_VALUES = [5, 10]

# Ragas LLM-судья для финального ответа.
#
# qwen: локальный Qwen3-14B через transformers.
# gigachat: GigaChat через LangChain ChatModel.
RAGAS_ENABLED = True
RAGAS_BACKEND = "custom"  # "custom" - простой надежный judge; "ragas" - библиотека ragas.
RAGAS_JUDGE_PROVIDER = "gigachat"
RAGAS_EMBEDDINGS_PROVIDER = "bge_m3"
RAGAS_CONTEXT_SOURCE = "retriever"
RAGAS_TIMEOUT_SECONDS = 20
RAGAS_MAX_WORKERS = 1
# Для RAGAS_BACKEND = "custom" это количество retry при невалидном JSON от GigaChat.
RAGAS_MAX_RETRIES = 3
RAGAS_METRICS = [
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "answer_similarity",
    "context_precision",
    "context_recall",
]

# Локальные embeddings для RAGAS.
#
# BGE-M3 нужен метрикам RAGAS, которые считают семантическую близость
# (например, answer_similarity/answer_relevancy и non-LLM context-метрики).
BGE_M3_MODEL_PATH = "/home/vladislav/models/bge-m3"
BGE_M3_LOCAL_FILES_ONLY = True
BGE_M3_DEVICE = "cpu"
BGE_M3_NORMALIZE_EMBEDDINGS = True

# Локальный Qwen через transformers.
#
# В закрытом контуре указывай только локальный путь.
# QWEN_LOCAL_FILES_ONLY = True запрещает transformers скачивать модель из интернета.
QWEN_PROVIDER = "qwen_transformers"
QWEN_MODEL_PATH = "models/Qwen3-14B"
QWEN_TASK = "text-generation"
QWEN_DEVICE = None
QWEN_DEVICE_MAP = "auto"
QWEN_TORCH_DTYPE = "auto"
QWEN_LOCAL_FILES_ONLY = True
QWEN_TIMEOUT_SECONDS = 60
QWEN_TEMPERATURE = 0
QWEN_MAX_NEW_TOKENS = 1024
QWEN_DO_SAMPLE = False
QWEN_RETURN_FULL_TEXT = False

# --- GigaChat: текущий домашний контур ---
# Для langchain-gigachat указывается базовый API URL, без /chat/completions.
# GIGACHAT_ACCESS_TOKEN - это временный access_token из OAuth-ответа,
# не Authorization key из личного кабинета.
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_ACCESS_TOKEN = os.getenv("GIGACHAT_ACCESS_TOKEN", "")
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0
GIGACHAT_VERIFY_SSL = False

# --- GigaChat: рабочий изолированный контур ---
# Когда будешь запускать в рабочем контуре, закомментируй домашний блок выше
# и раскомментируй этот блок, если там используются другие адрес/модель/токен.
#
# GIGACHAT_BASE_URL = ""
# GIGACHAT_ACCESS_TOKEN = ""
# GIGACHAT_MODEL = "GigaChat"
# GIGACHAT_TEMPERATURE = 0
# GIGACHAT_VERIFY_SSL = True

# Ограничение тестовой системы для GigaChat-судьи.
# Это не параметр GigaChat API. Нужно для режима 1 запрос раз в 20 секунд.
# RAGAS_MAX_WORKERS выше должен оставаться 1.
GIGACHAT_MIN_SECONDS_BETWEEN_REQUESTS = 20
