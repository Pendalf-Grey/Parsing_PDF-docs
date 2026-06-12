from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import orjson
import psycopg
from psycopg.types.json import Jsonb


DEFAULT_INPUT = Path("out/chunked_english_text_tables.jsonl")
DEFAULT_SCHEMA = Path("sql/001_document_chunks_pgvector.sql")


def database_url() -> str:
    """Возвращает URL подключения к PostgreSQL из окружения."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError("DATABASE_URL is required, for example postgresql://user:pass@localhost:5432/db")

    return url


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Читает записи чанков из JSONL."""
    records: list[dict[str, Any]] = []

    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))

    return records


def apply_schema(conn: psycopg.Connection, schema_path: Path) -> None:
    """Применяет схему pgvector и индексы."""
    sql = schema_path.read_text(encoding="utf-8")

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


def chunk_params(record: dict[str, Any]) -> tuple[Any, ...]:
    """Преобразует одну JSON-запись чанка в параметры SQL."""
    return (
        record["chunk_id"],
        record["doc_id"],
        record["source_file"],
        record["chunk_index"],
        record.get("page_start"),
        record.get("page_end"),
        record.get("section_path") or [],
        record["text"],
        record["word_count"],
        record.get("source_record_indices") or [],
        record.get("source_content_types") or [],
        Jsonb(record),
    )


def upsert_chunks(conn: psycopg.Connection, records: list[dict[str, Any]]) -> None:
    """Вставляет или обновляет строки чанков, сохраняя уже рассчитанные embeddings."""
    sql = """
        INSERT INTO document_chunks (
            chunk_id,
            doc_id,
            source_file,
            chunk_index,
            page_start,
            page_end,
            section_path,
            text,
            word_count,
            source_record_indices,
            source_content_types,
            raw_metadata
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s::text[],
            %s,
            %s,
            %s::integer[],
            %s::text[],
            %s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            source_file = EXCLUDED.source_file,
            chunk_index = EXCLUDED.chunk_index,
            page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end,
            section_path = EXCLUDED.section_path,
            text = EXCLUDED.text,
            word_count = EXCLUDED.word_count,
            source_record_indices = EXCLUDED.source_record_indices,
            source_content_types = EXCLUDED.source_content_types,
            raw_metadata = EXCLUDED.raw_metadata,
            updated_at = now()
    """

    with conn.cursor() as cur:
        cur.executemany(sql, [chunk_params(record) for record in records])

    conn.commit()


def count_chunks(conn: psycopg.Connection) -> tuple[int, int]:
    """Возвращает общее число чанков и число чанков с embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                count(*)::integer,
                count(embedding)::integer
            FROM document_chunks
            """
        )
        total, embedded = cur.fetchone()

    return int(total), int(embedded)


def main() -> None:
    """Загружает JSONL с чанками в PostgreSQL/pgvector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSONL-файл с чанками")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Путь к SQL-схеме")
    parser.add_argument("--skip-schema", action="store_true", help="Не применять схему перед загрузкой")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))

    with psycopg.connect(database_url()) as conn:
        if not args.skip_schema:
            apply_schema(conn, Path(args.schema))

        upsert_chunks(conn, records)
        total, embedded = count_chunks(conn)

    print(f"loaded chunks from jsonl: {len(records)}")
    print(f"chunks in postgres: {total}")
    print(f"chunks with embeddings: {embedded}")


if __name__ == "__main__":
    main()
