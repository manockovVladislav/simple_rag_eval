# Local RAG Eval

Локальная система оценки RAG pipeline на золотых вопросах.

Что делает:

1. Читает золотой набор из `data/golden_questions.xlsx`.
2. Прогоняет вопросы через ваш pipeline.
3. Сохраняет ответы, контексты retriever и reranker в уникальный `outputs/runs/rag_run_*.xlsx`.
4. Считает метрики и дописывает результат в один файл `outputs/metrics/metrics_summary.xlsx`.

## Установка

```bash
pip install -e .
```

## Конфиг

Все настройки в [config.py](config.py). Формат конфига плоский, без словарей:

```python
PIPELINE_FACTORY = "my_project.my_rag:create_pipeline"

RAGAS_JUDGE_PROVIDER = "qwen"
RAGAS_TIMEOUT_SECONDS = 60
RAGAS_MAX_WORKERS = 1

QWEN_PROVIDER = "qwen_transformers"
QWEN_MODEL_PATH = "/models/qwen3"
QWEN_DEVICE_MAP = "auto"
QWEN_TORCH_DTYPE = "auto"
```

`qwen` используется локально через `transformers`, без HTTP API. Модель должна быть доступна по `QWEN_MODEL_PATH` или как установленный model id.

`gigachat` используется через API. Для него настройте:

```python
RAGAS_JUDGE_PROVIDER = "gigachat"
GIGACHAT_AUTH_TYPE = "bearer_env"
GIGACHAT_API_KEY_ENV = "GIGACHAT_API_KEY"
```

## Золотой Файл

По умолчанию: `data/golden_questions.xlsx`.

Минимальные колонки:

- `question` - вопрос.
- `expected_answer` - эталонный ответ для answer/Ragas метрик.
- `expected_context_ids` - id релевантных документов для deterministic retrieval метрик.

Названия колонок меняются в `config.py`:

```python
QUESTION_COLUMN = "question"
EXPECTED_ANSWER_COLUMN = "expected_answer"
EXPECTED_CONTEXT_IDS_COLUMN = "expected_context_ids"
```

## Pipeline Adapter

В `config.py` укажите фабрику:

```python
PIPELINE_FACTORY = "my_project.my_rag:create_pipeline"
```

Фабрика должна вернуть объект с методом `answer_question(question: str)`.

Ожидаемый формат результата:

```python
{
    "answer": "...",
    "retriever_contexts": [
        {"id": "doc_1", "text": "...", "score": 0.91},
    ],
    "reranker_contexts": [
        {"id": "doc_1", "text": "...", "score": 0.97},
    ],
    "metadata": {},
}
```

Если retriever, reranker или answer отсутствует, верните пустой список или `None`; метрики для отсутствующей части не считаются.

## Запуск

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

## Итоговый XLSX

Все метрики пишутся только в:

```text
outputs/metrics/metrics_summary.xlsx
```

Листы:

- `summary` - одна строка на каждый прогон.
- `details` - строки по каждому вопросу.

В начале листов идут служебные поля и метрики судьи `ragas_*`, затем остальные метрики.

## Метрики

Ragas LLM judge:

- `ragas_retriever_context_precision`
- `ragas_retriever_context_recall`
- `ragas_reranker_context_precision`
- `ragas_reranker_context_recall`
- `ragas_faithfulness`
- `ragas_answer_correctness`

Ragas работает последовательно:

```python
RAGAS_MAX_WORKERS = 1
RAGAS_TIMEOUT_SECONDS = 60
RAGAS_MAX_RETRIES = 0
```

Deterministic retrieval метрики по `expected_context_ids`:

- `precision@k`
- `recall@k`
- `hit@k`
- `mrr@k`
- `ndcg@k`

Локальные answer метрики:

- `answer_exact_match`
- `answer_contains_expected`
- `answer_token_f1`

## Проверка Без Загрузки Qwen

Чтобы проверить pipeline и запись Excel без загрузки transformers-модели, временно поставьте:

```python
RAGAS_ENABLED = False
```

После проверки включите обратно:

```python
RAGAS_ENABLED = True
```
