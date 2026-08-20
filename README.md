# Local RAG Eval

Локальная система проверки retriever / reranker и финального RAG-ответа в закрытом контуре.

Что делает:

1. Читает вопросы из `data/golden_questions.xlsx`.
2. Запускает выбранный pipeline из `config.py`.
3. После retriever/reranker отправляет один и тот же контекст в одну модель или асинхронно в GigaChat, Qwen и GLM.
4. Сохраняет ответы в три независимых JSON: `*_gigachat.json`, `*_qwen.json`, `*_glm.json`.
5. Судья последовательно читает сохранённый JSON каждой модели и считает её метрики.
6. Сохраняет широкую сравнительную таблицу в `outputs/metrics/model_comparison.xlsx`.

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
- `ground_truth` - эталонный ответ для проверки финального ответа.

Опциональная колонка для retrieval-метрик:

- `chunk_id` - один или несколько релевантных chunk id через `;`, `,`, `|` или JSON-list.

Если `chunk_id` заполнен, система считает top-k метрики для `retriever_contexts` и `reranker_contexts`.
Если `chunk_id` не заполнен, retrieval-метрики пропускаются, а judge продолжает оценивать финальный ответ.
Для совместимости старое имя `expected_answer` остается fallback для `ground_truth`.

Названия колонок меняются в `config.py`:

```python
QUESTION_COLUMN = "question"
GROUND_TRUTH_COLUMN = "ground_truth"
CHUNK_ID_COLUMN = "chunk_id"
```

## Два Ретривера

Внешний репозиторий `oaofr_assistant_gr` клонируется в корень проекта. Его код не меняем.

Обертки для тестовой системы лежат здесь:

```text
src/rag_eval/oaofr_assistant_pipeline.py
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

Для файла `hybrid_retriever_new_all_formulas.py`:

```python
PIPELINE_FACTORY = "rag_eval.oaofr_assistant_pipeline:create_retriever_pipeline"
```

Для файла `hybrid_retriever_new_all_formulas_and_reranker.py`:

```python
PIPELINE_FACTORY = "rag_eval.oaofr_assistant_pipeline:create_reranker_pipeline"
```

Параметры создания класса добавляются в `RETRIEVER_INIT_KWARGS`:

```python
RETRIEVER_INIT_KWARGS = {
    "faiss_path": "oaofr_assistant_gr/vectors/index.faiss",
    "pkl_path": "oaofr_assistant_gr/vectors/metadata.pkl",
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

После этого система берет контексты из `RAG_CONTEXT_SOURCE` (`reranker` по умолчанию, fallback на `retriever`)
и отправляет их в модель из `RAG_LLM_PROVIDER`.

```python
RAG_ANSWER_ENABLED = True
RAG_LLM_PROVIDER = "gigachat"
RAG_PARALLEL_GENERATION_ENABLED = True
RAG_LLM_PROVIDERS = ["gigachat", "qwen", "glm"]
RAG_GENERATION_MAX_WORKERS = 3
RAG_CONTEXT_SOURCE = "reranker"
RAG_MAX_CONTEXTS = 10
```

`RAG_LLM_PROVIDER` — основная модель: её ответ дублируется в старую колонку `answer` для обратной
совместимости. Ответы трёх моделей пишутся в `answer_gigachat`, `answer_qwen`, `answer_glm`.
Ошибка одного API сохраняется в `error_<provider>` и не отменяет два остальных ответа.

Старый режим включается одной строкой:

```python
RAG_PARALLEL_GENERATION_ENABLED = False
```

Судья настраивается отдельно:

```python
RAGAS_BACKEND = "custom"
RAGAS_JUDGE_PROVIDER = "judge"

JUDGE_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
JUDGE_ACCESS_TOKEN = "..."
JUDGE_MODEL = "GigaChat"
JUDGE_TIMEOUT_SECONDS = 20
JUDGE_TEMPERATURE = 0
JUDGE_VERIFY_SSL = False
JUDGE_MIN_SECONDS_BETWEEN_REQUESTS = 20
```

Повторы запросов отвечающих моделей задаются отдельно. Значение `3` означает не более трёх
полных попыток, включая первый запрос:

```python
GIGACHAT_MAX_ATTEMPTS = 3
QWEN_MAX_ATTEMPTS = 3
GLM_MAX_ATTEMPTS = 3
```

## GigaChat, Qwen и GLM

Все три модели вызываются через один GigaChat-compatible API. Локального Qwen в коде нет:

```python
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_ACCESS_TOKEN = "..."
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0

QWEN_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
QWEN_ACCESS_TOKEN = "..."
QWEN_MODEL = "Qwen"
QWEN_TEMPERATURE = 0

GLM_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GLM_ACCESS_TOKEN = "..."
GLM_MODEL = "GLM"
GLM_TEMPERATURE = 0
```

Ограничение закрытого контура:

```python
RAGAS_MAX_WORKERS = 1
JUDGE_MIN_SECONDS_BETWEEN_REQUESTS = 20
RAGAS_JUDGE_MAX_CONTEXT_CHARS = 60000
```

`JUDGE_MIN_SECONDS_BETWEEN_REQUESTS` - это локальный rate limit тестовой системы, не параметр GigaChat API.
Полные контексты не обрезаются в run JSON. `RAGAS_JUDGE_MAX_CONTEXT_CHARS` ограничивает
только копию, передаваемую судье, чтобы не превысить окно модели.

`RAGAS_BACKEND = "custom"` включает простой локальный judge: GigaChat оценивает LLM-метрики по строгому JSON,
а `answer_similarity` считается через embeddings из `RAGAS_EMBEDDINGS_PROVIDER`.
Custom judge возвращает score по десятичной шкале с шагом `0.1`; лимит на длину его ответа не задаётся.
`RAGAS_BACKEND = "ragas"` оставляет библиотечный Ragas.

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
outputs/runs/rag_run_*_gigachat.json
outputs/runs/rag_run_*_qwen.json
outputs/runs/rag_run_*_glm.json
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
outputs/runs/rag_run_*_gigachat.json
outputs/runs/rag_run_*_qwen.json
outputs/runs/rag_run_*_glm.json
```

Итоговые метрики:

```text
outputs/metrics/metrics_summary.xlsx
outputs/metrics/metrics_summary.json
outputs/metrics/model_comparison.xlsx
```

В `model_comparison.xlsx` одна строка соответствует одному вопросу. Сначала идут `date`, `question`,
`context`, `correct_answer`, затем для каждой модели её `answer_<model>` и колонки
`metric_<model>_*`.

JSON — полный источник данных без лимита длины ячейки Excel. В run-Excel есть отдельная
колонка `json_file_<provider>` для каждой модели. При расчёте метрик судья по очереди читает эти JSON.
Каждый run также хранит snapshot параметров. В `metrics_summary` выводятся `model_name`, `temperature`,
`judge_model_name`, `judge_temperature`, `k_rrf`, `fusion_method`, `alpha`, `bias`, `rerank_initial_k`,
`retriever_top_k`, а также средние `retriever_context_count_mean` и `reranker_context_count_mean`.

`ragas_judge_score_mean` — равновесное среднее по всем доступным score-метрикам судьи. Сначала для
каждой метрики считается среднее по вопросам без пустых значений, затем эти средние усредняются
с одинаковым весом. Колонки `*_reason`, `*_evidence`, `*_raw` и `*_error` в расчёт не входят. В мультимодельном
режиме также считаются `ragas_<provider>_judge_score_mean` отдельно для GigaChat, Qwen и GLM.
В `main.ipynb` есть отключённая по умолчанию ячейка grid search; каждая комбинация создаёт отдельный run.

Листы:

- `summary` - агрегаты по прогону.
- `details` - компактные метрики по каждому вопросу без технических raw-ответов.
- `judge` - только judge score по каждому вопросу.
- `questions` - вопрос, ответ, эталон, контексты, ошибки, причины и evidence по judge-метрикам.
- `judge_debug` - технические raw-ответы судьи и ошибки парсинга/вызова.
- `retrieval` - retrieval-контексты и top-k метрики.

В мультимодельном режиме на листе `details` рядом с каждым вопросом появляются
`answer_<provider>`, локальные `answer_<provider>_*` и оценки неизменённого GigaChat-судьи
`ragas_<provider>_*`.

Основные метрики:

- `retriever_hit_rate_at_10`
- `retriever_recall_at_10`
- `retriever_precision_at_10`
- `retriever_mrr_at_10`
- `retriever_ndcg_at_10`
- `reranker_hit_rate_at_10`
- `reranker_recall_at_10`
- `reranker_precision_at_10`
- `reranker_mrr_at_10`
- `reranker_ndcg_at_10`
- `ragas_faithfulness`
- `ragas_answer_correctness`
- `ragas_answer_relevancy`
- `ragas_answer_similarity`
- `answer_exact_match`
- `answer_contains_expected`
- `answer_token_f1`

Если в golden-файле нет `chunk_id`, retrieval-метрики пропускаются. Судья оценивает только финальный `answer`.

Подробный лог пишется в:

```text
outputs/logs/rag_eval.log
```

Туда попадают вопрос, найденные контексты, запрос к RAG LLM, ответ модели и результаты judge-метрик.
