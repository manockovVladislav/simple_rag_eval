# Local RAG Eval

Локальная система проверки retriever / reranker в закрытом контуре.

Что делает:

1. Читает вопросы из `data/golden_questions.xlsx`.
2. Запускает выбранный pipeline из `config.py`.
3. Сохраняет ответы и контексты в `outputs/runs/rag_run_*.xlsx`.
4. Считает метрики и дописывает результат в `outputs/metrics/metrics_summary.xlsx`.

## Установка

```bash
pip install -e .
```

Зависимости ставятся локально. Интернет во время работы не нужен, если модели, индексы и Python-пакеты уже есть в среде.

## Golden-Файл

По умолчанию используется:

```text
data/golden_questions.xlsx
```

Минимальные колонки:

- `question` - вопрос.
- `expected_answer` - эталонный ответ для Ragas/context метрик.

Названия колонок меняются в `config.py`:

```python
QUESTION_COLUMN = "question"
EXPECTED_ANSWER_COLUMN = "expected_answer"
```

## Два Ретривера

Внешний репозиторий `retriver_gr_oaofr` клонируется в корень проекта. Его код не меняем.

Обертки для тестовой системы лежат здесь:

```text
src/rag_eval/retriver_gr_oaofr_pipeline.py
```

Там два класса:

- `RagEvalHybridRetrieverOnly` наследуется от `HybridRetriever_all_formulas`.
- `RagEvalHybridRetrieverWithReranker` наследуется от `HybridRetriever_all_formulas_and_reranker`.

Оба класса добавляют метод `answer_question(question)` и возвращают `PipelineResult`, который понимает тестовая система.

## Что Выбрать В config.py

Для демо без внешнего retriever:

```python
PIPELINE_FACTORY = "examples.my_pipeline:create_pipeline"
```

Для файла `hibrid_retriver_all_formulas.py`:

```python
PIPELINE_FACTORY = "rag_eval.retriver_gr_oaofr_pipeline:create_retriever_pipeline"
```

Для файла `hibrid_retriver_all_formulas_and_reranker.py`:

```python
PIPELINE_FACTORY = "rag_eval.retriver_gr_oaofr_pipeline:create_reranker_pipeline"
```

Параметры создания класса добавляются в `RETRIEVER_INIT_KWARGS`:

```python
RETRIEVER_INIT_KWARGS = {
    "faiss_path": "retriver_gr_oaofr/vectors/index.faiss",
    "pkl_path": "retriver_gr_oaofr/vectors/corpus.pkl",
    "model_name": "bge-m3",
    "fusion_method": "rrf",
    "reranker": True,
}
RETRIEVER_TOP_K = 10
```

Для retriever-only заполняется `retriever_contexts`.

Для retriever + reranker система делает два поиска:

- без `self.reranker` для `retriever_contexts`;
- со штатным `self.reranker` для `reranker_contexts`.

## Qwen И GigaChat

Qwen запускается только локально через `transformers`:

```python
RAGAS_JUDGE_PROVIDER = "qwen"
QWEN_MODEL_PATH = "models/Qwen3-14B"
QWEN_LOCAL_FILES_ONLY = True
```

GigaChat получает только четыре параметра:

```python
RAGAS_JUDGE_PROVIDER = "gigachat"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_ACCESS_TOKEN = "..."
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0
```

Ограничение закрытого контура:

```python
RAGAS_MAX_WORKERS = 1
GIGACHAT_MIN_SECONDS_BETWEEN_REQUESTS = 10
```

`GIGACHAT_MIN_SECONDS_BETWEEN_REQUESTS` - это локальный rate limit тестовой системы, не параметр GigaChat API.

## Быстрая Проверка

Сначала проверьте pipeline без загрузки Qwen/GigaChat:

```python
MAX_QUESTIONS = 1
RAGAS_ENABLED = False
```

Запуск:

```bash
python scripts/run_questions.py --config config.py
```

После этого появится файл:

```text
outputs/runs/rag_run_*.xlsx
```

## Полный Запуск

Включите:

```python
MAX_QUESTIONS = None
RAGAS_ENABLED = True
```

Прогнать вопросы:

```bash
python scripts/run_questions.py --config config.py
```

Посчитать метрики по последнему run-файлу:

```bash
python scripts/calc_metrics.py --config config.py
```

Посчитать метрики по конкретному run-файлу:

```bash
python scripts/calc_metrics.py --config config.py --run-file outputs/runs/rag_run_YYYYMMDD_HHMMSS_xxxxxx.xlsx
```

То же из Jupyter:

```python
from rag_eval import EvaluationRunner, MetricsCalculator

run_file = EvaluationRunner.from_config("config.py").run()
metrics_file = MetricsCalculator.from_config("config.py").evaluate(run_file)
```

Корневой notebook:

```text
main.ipynb
```

## Результаты

Run-файлы:

```text
outputs/runs/rag_run_*.xlsx
```

Итоговые метрики:

```text
outputs/metrics/metrics_summary.xlsx
```

Листы:

- `summary` - агрегаты по прогону.
- `details` - метрики по каждому вопросу.

Основные метрики:

- `ragas_retriever_context_precision`
- `ragas_retriever_context_recall`
- `ragas_reranker_context_precision`
- `ragas_reranker_context_recall`
- `ragas_faithfulness`
- `ragas_answer_correctness`
- `answer_exact_match`
- `answer_contains_expected`
- `answer_token_f1`

Если pipeline не генерирует answer, answer-метрики пропускаются. Context-метрики считаются по `question`, `expected_answer` и найденным контекстам.
