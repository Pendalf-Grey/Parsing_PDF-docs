# Parsing PDF Docs

Проект извлекает из PDF англоязычный текст, заголовки и таблицы через Docling, а затем готовит overlapped JSONL-чанки для следующего этапа embeddings/RAG.

## Что получается на выходе

Первый этап, `parce_docling_english.py`, создает структурированные Docling-записи:

- `out/parsed_english_text_tables.jsonl`
- `out/parsed_english_text_tables.debug.md`

Второй этап, `chunk_docling_jsonl.py`, читает JSONL первого этапа и создает чанки с overlap:

- `out/chunked_english_text_tables.jsonl`

## Установка

Нужен Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Скопируйте пример переменных окружения:

```bash
cp .env.example .env
source .env
```

## Запуск парсера PDF

Обычный запуск для PDF-файлов из папки `data/pdfs`:

```bash
python parce_docling_english.py --input data/pdfs --out out
```

Для одного PDF:

```bash
python parce_docling_english.py --input "data/pdfs/HP ProLiant DL360 G6 Server Maintenance and Service Guide.pdf" --out out
```

Если PDF сканированный, включите OCR:

```bash
python parce_docling_english.py --input data/pdfs --out out --ocr
```

## Запуск на Mac GPU

Для Apple Silicon можно попробовать MPS:

```bash
python parce_docling_english.py --input data/pdfs --out out --device mps
```

Если PyTorch в текущем окружении не видит Apple GPU, скрипт предупредит об этом, и Docling может вернуться к CPU. Универсальный режим по умолчанию:

```bash
python parce_docling_english.py --input data/pdfs --out out --device auto
```

## Чанкование с overlap

После парсинга запустите второй этап:

```bash
python chunk_docling_jsonl.py --input out/parsed_english_text_tables.jsonl --out out/chunked_english_text_tables.jsonl
```

По умолчанию:

- размер чанка: `180` слов
- overlap: `40` слов
- таблицы включены в чанки как Markdown-текст

Настроить размер чанков:

```bash
python chunk_docling_jsonl.py \
  --input out/parsed_english_text_tables.jsonl \
  --out out/chunked_english_text_tables.jsonl \
  --chunk-words 220 \
  --overlap-words 50
```

Исключить таблицы:

```bash
python chunk_docling_jsonl.py --no-tables
```

## Формат parsed JSONL

Каждая строка в `parsed_english_text_tables.jsonl` — это одна запись, извлеченная Docling: заголовок, текстовый блок или таблица. Все текстовые значения нормализуются и приводятся к нижнему регистру.

Пример текстовой записи:

```json
{
  "doc_id": "79a8dcc27ec4fece3ead995fde6ae36eaa925d5ab39ea93ff0117440cf3d2901",
  "source_file": "HP ProLiant DL360 G6 Server Maintenance and Service Guide.pdf",
  "content_type": "text",
  "page_start": 5,
  "page_end": 5,
  "section_path": ["customer self repair"],
  "text": "mandatory -parts for which customer self repair is mandatory...",
  "table_markdown": null,
  "table_json": null
}
```

Пример записи таблицы:

```json
{
  "doc_id": "79a8dcc27ec4fece3ead995fde6ae36eaa925d5ab39ea93ff0117440cf3d2901",
  "source_file": "HP ProLiant DL360 G6 Server Maintenance and Service Guide.pdf",
  "content_type": "table",
  "page_start": 16,
  "page_end": 16,
  "section_path": ["customer self repair"],
  "text": "item description spare part number customer self repair...",
  "table_markdown": "|   item | description | spare part number | customer self repair (on page 5)   |",
  "table_json": [
    {
      "item": "1",
      "description": "access panel",
      "spare part number": "532146-001",
      "customer self repair (on page 5)": "mandatory 1"
    }
  ]
}
```

Метаданные parsed-записи:

- `doc_id` — SHA-256 хеш исходного PDF. Используется как стабильный идентификатор документа.
- `source_file` — имя PDF-файла, из которого была получена запись.
- `content_type` — тип записи: `heading`, `text` или `table`.
- `page_start` — первая страница, к которой Docling привязал элемент.
- `page_end` — последняя страница элемента. Для обычных блоков чаще совпадает с `page_start`.
- `section_path` — текущий путь по заголовкам документа. Помогает понять, в каком разделе находится запись.
- `text` — основной текст записи. Для таблиц это плоское текстовое представление заголовков и ячеек.
- `table_markdown` — Markdown-представление таблицы. Заполнено только для `content_type = "table"`.
- `table_json` — таблица как список словарей, где ключи — названия колонок. Заполнено только для `content_type = "table"`.

## Формат chunked JSONL

Каждая строка в `chunked_english_text_tables.jsonl` — это один чанк для embeddings/RAG. Чанки собираются внутри одного документа и одного `section_path`; соседние чанки внутри секции имеют overlap.

```json
{
  "chunk_id": "d7defb77b439d5b2f3766a039065f7afcffdbb20721a1ac588d33125e3493c8c",
  "doc_id": "79a8dcc27ec4fece3ead995fde6ae36eaa925d5ab39ea93ff0117440cf3d2901",
  "source_file": "HP ProLiant DL360 G6 Server Maintenance and Service Guide.pdf",
  "chunk_index": 1,
  "page_start": 5,
  "page_end": 24,
  "section_path": ["customer self repair"],
  "text": "customer self repair hp products are designed with many customer self repair...",
  "word_count": 180,
  "source_record_indices": [5, 6, 7, 8, 9],
  "source_content_types": ["heading", "table", "text"]
}
```

Метаданные chunked-записи:

- `chunk_id` — SHA-256 идентификатор чанка, рассчитанный из `doc_id`, `source_file`, номера чанка и текста.
- `doc_id` — тот же идентификатор документа, что и в parsed JSONL.
- `source_file` — исходный PDF-файл.
- `chunk_index` — порядковый номер чанка в выходном JSONL.
- `page_start` — минимальная страница среди исходных parsed-записей, вошедших в чанк.
- `page_end` — максимальная страница среди исходных parsed-записей, вошедших в чанк.
- `section_path` — раздел документа, внутри которого собран чанк.
- `text` — итоговый текст чанка для embeddings. В него могут входить заголовки, текстовые блоки и Markdown-представления таблиц.
- `word_count` — количество слов в чанке.
- `source_record_indices` — индексы исходных строк из parsed JSONL, использованных при сборке чанка.
- `source_content_types` — типы исходных записей, вошедших в чанк: например `heading`, `text`, `table`.

`text` в чанках уже готов для следующего шага: расчета embeddings и загрузки в PostgreSQL/pgvector или другую векторную базу.

## PostgreSQL и pgvector

Локальную базу с pgvector можно поднять через Docker:

```bash
docker compose up -d postgres
```

Стандартный `DATABASE_URL` из `.env.example`:

```bash
postgresql://parsing_pdf:parsing_pdf@localhost:5432/parsing_pdf
```

Схема создается из файла:

```bash
sql/001_document_chunks_pgvector.sql
```

Она создает таблицу `document_chunks` с `embedding vector(4096)` под `Qwen3-Embedding-8B`, а также индексы:

- обычные B-tree индексы по `doc_id`, `source_file`, `chunk_index`, страницам
- GIN индексы по `section_path`, `source_content_types`, `raw_metadata`
- full-text GIN индекс по `text`
- HNSW индекс по первым 2000 измерениям `embedding` через `subvector(...)`

Почему subvector: `Qwen3-Embedding-8B` возвращает `4096` измерений, а pgvector HNSW для типа `vector` индексирует до `2000` измерений. Поэтому PostgreSQL сначала быстро выбирает кандидатов по `subvector(embedding, 1, 2000)`, а `search_pgvector_mlx.py` затем rerank-ит кандидатов по полному `4096`-мерному embedding.

## Загрузка чанков в PostgreSQL

После создания `out/chunked_english_text_tables.jsonl` загрузите чанки:

```bash
python ingest_chunks_to_postgres.py \
  --input out/chunked_english_text_tables.jsonl
```

Скрипт:

- применяет SQL-схему
- делает upsert по `chunk_id`
- сохраняет все метаданные чанка
- оставляет `embedding = NULL`, если вектор еще не посчитан

## Скачивание Qwen3 Embedding для Mac

Модель не коммитится в git и хранится локально в `models/`.

```bash
hf download mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --local-dir models/Qwen3-Embedding-8B-4bit-DWQ
```

Для Apple Silicon используется MLX-модель:

```bash
QWEN3_EMBEDDING_MODEL=models/Qwen3-Embedding-8B-4bit-DWQ
```

## Расчет embeddings и запись в pgvector

После загрузки чанков в PostgreSQL запустите:

```bash
python embed_chunks_pgvector_mlx.py \
  --model models/Qwen3-Embedding-8B-4bit-DWQ \
  --batch-size 2
```

Скрипт:

- берет строки из `document_chunks`, где `embedding IS NULL`
- считает embeddings через `mlx-embeddings`
- проверяет размерность `4096`
- L2-нормализует вектор
- записывает результат в `embedding vector(4096)`
- сохраняет `embedding_model` и `embedding_created_at`

Если нужно сохранить сырые ненормализованные vectors:

```bash
python embed_chunks_pgvector_mlx.py --no-normalize
```

## Проверка semantic search

Когда embeddings записаны, можно проверить поиск:

```bash
python search_pgvector_mlx.py "how to replace hot-plug power supply" --limit 5
```

Запрос тоже эмбеддится той же MLX Qwen3-моделью, затем PostgreSQL ищет ближайшие чанки через cosine distance:

```sql
ORDER BY embedding <=> query_embedding
```
