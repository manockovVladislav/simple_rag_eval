# Paths
GOLDEN_QUESTIONS_PATH = "data/golden_questions.xlsx"
RUN_OUTPUTS_DIR = "outputs/runs"
METRICS_OUTPUTS_DIR = "outputs/metrics"
SUMMARY_METRICS_FILE = "outputs/metrics/metrics_summary.xlsx"

# Golden dataset columns
QUESTION_COLUMN = "question"
EXPECTED_ANSWER_COLUMN = "expected_answer"

# Your RAG pipeline adapter
PIPELINE_FACTORY = "examples.my_pipeline:create_pipeline"

# Run settings
MAX_QUESTIONS = None
SLEEP_SECONDS = 0

# Local answer metrics
ANSWER_METRICS_ENABLED = True

# Ragas LLM judge
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

# Local Qwen through transformers. Use local path or HuggingFace model id.
QWEN_PROVIDER = "qwen_transformers"
QWEN_MODEL_PATH = "Qwen/Qwen3-8B"
QWEN_TASK = "text-generation"
QWEN_DEVICE = None
QWEN_DEVICE_MAP = "auto"
QWEN_TORCH_DTYPE = "auto"
QWEN_TIMEOUT_SECONDS = 60
QWEN_TEMPERATURE = 0
QWEN_MAX_NEW_TOKENS = 1024
QWEN_DO_SAMPLE = False
QWEN_RETURN_FULL_TEXT = False

# GigaChat API
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_AUTH_TYPE = "bearer_env"
GIGACHAT_API_KEY_ENV = "GIGACHAT_API_KEY"
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_TIMEOUT_SECONDS = 60
GIGACHAT_TEMPERATURE = 0
GIGACHAT_VERIFY_SSL = True
