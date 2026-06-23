# Пути
GOLDEN_QUESTIONS_PATH = "data/golden_questions.xlsx"
RUN_OUTPUTS_DIR = "outputs/runs"
METRICS_OUTPUTS_DIR = "outputs/metrics"
SUMMARY_METRICS_FILE = "outputs/metrics/metrics_summary.xlsx"

# Колонки в golden-файле
QUESTION_COLUMN = "question"
EXPECTED_ANSWER_COLUMN = "expected_answer"

# Какой pipeline запускать.
#
# 1. Демо-проверка тестовой системы без внешнего ретривера:
PIPELINE_FACTORY = "examples.my_pipeline:create_pipeline"
#
# 2. Твой файл retriver_gr_oaofr/retriver/hibrid_retriver_all_formulas.py
#    Только retriever, без reranker. Результаты попадут в retriever_contexts.
# PIPELINE_FACTORY = "rag_eval.retriver_gr_oaofr_pipeline:create_retriever_pipeline"
#
# 3. Твой файл retriver_gr_oaofr/retriver/hibrid_retriver_all_formulas_and_reranker.py
#    Retriever + reranker. Результаты retriever попадут в retriever_contexts,
#    результаты reranker попадут в reranker_contexts.
# PIPELINE_FACTORY = "rag_eval.retriver_gr_oaofr_pipeline:create_reranker_pipeline"

# Параметры создания твоих классов.
#
# Сюда добавлять ровно те аргументы, которые принимает __init__ класса:
# - faiss_path: путь к локальному FAISS индексу;
# - pkl_path: путь к локальному pkl с корпусом/метаданными;
# - model_name: "bge-m3" или "e5-large";
# - fusion_method, alpha, bias, device и другие параметры твоего класса;
# - reranker: True/False только для класса retriever + reranker.
RETRIEVER_INIT_ARGS = []
RETRIEVER_INIT_KWARGS = {
    # "faiss_path": "retriver_gr_oaofr/vectors/index.faiss",
    # "pkl_path": "retriver_gr_oaofr/vectors/corpus.pkl",
    # "model_name": "bge-m3",
    # "fusion_method": "rrf",
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

# Локальные answer-метрики без LLM-судьи
ANSWER_METRICS_ENABLED = True

# Ragas LLM-судья.
#
# qwen: локальный Qwen3-14B через transformers.
# gigachat: GigaChat через LangChain ChatModel.
RAGAS_ENABLED = True
RAGAS_JUDGE_PROVIDER = "qwen"
RAGAS_CONTEXT_SOURCE = "reranker"
RAGAS_TIMEOUT_SECONDS = 60
RAGAS_MAX_WORKERS = 1
RAGAS_MAX_RETRIES = 0
RAGAS_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

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

# GigaChat.
#
# Для GigaChat передаются только эти 4 параметра:
# base_url, access_token, model, temperature.
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_ACCESS_TOKEN = ""
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0

# Ограничение тестовой системы для GigaChat-судьи.
# Это не параметр GigaChat API. Нужно для режима 1 запрос раз в 10 секунд.
# RAGAS_MAX_WORKERS выше должен оставаться 1.
GIGACHAT_MIN_SECONDS_BETWEEN_REQUESTS = 10
