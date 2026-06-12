from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Sequence
from typing import Any

import psycopg


DEFAULT_API_URL = "http://localhost:8001/embed"
DEFAULT_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 8


def database_url() -> str:
    """Возвращает URL подключения к PostgreSQL из окружения."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError("DATABASE_URL is required, for example postgresql://user:pass@localhost:5432/db")

    return url


def fetch_pending_chunks(conn: psycopg.Connection, limit: int) -> list[tuple[str, str]]:
    """Получает чанки, для которых еще не рассчитаны embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, text
            FROM document_chunks
            WHERE embedding IS NULL
            ORDER BY source_file, chunk_index
            LIMIT %s
            """,
            (limit,),
        )
        return [(str(chunk_id), str(text)) for chunk_id, text in cur.fetchall()]


def vector_literal(vector: Sequence[float]) -> str:
    """Сериализует Python-последовательность float в формат литерала pgvector."""
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def call_embedding_worker(api_url: str, texts: list[str], batch_size: int) -> dict[str, Any]:
    """Отправляет тексты в локальный embedding-worker и возвращает JSON-ответ."""
    payload = json.dumps({
        "texts": texts,
        "normalize": True,
        "batch_size": batch_size,
    }).encode("utf-8")

    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def update_embeddings(
    conn: psycopg.Connection,
    rows: list[tuple[str, list[float]]],
    model_name_or_path: str,
) -> None:
    """Записывает embeddings обратно в PostgreSQL."""
    sql = """
        UPDATE document_chunks
        SET
            embedding = %s::vector,
            embedding_model = %s,
            embedding_created_at = now(),
            updated_at = now()
        WHERE chunk_id = %s
    """
    params = [
        (vector_literal(embedding), model_name_or_path, chunk_id)
        for chunk_id, embedding in rows
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, params)

    conn.commit()


def batched(items: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    """Разбивает чанки на батчи для HTTP worker-а."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def validate_response(data: dict[str, Any], expected_count: int, expected_dimension: int) -> list[list[float]]:
    """Проверяет ответ worker-а перед записью в pgvector."""
    embeddings = data.get("embeddings") or []
    dimension = data.get("dimension")
    count = data.get("count")

    if count != expected_count:
        raise ValueError(f"Expected count {expected_count}, got {count}")

    if dimension != expected_dimension:
        raise ValueError(f"Expected dimension {expected_dimension}, got {dimension}")

    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}")

    return embeddings


def embed_pending_chunks(
    conn: psycopg.Connection,
    api_url: str,
    batch_size: int,
    limit: int,
    expected_dimension: int,
) -> int:
    """Считает embeddings для ожидающих чанков через HTTP worker."""
    pending = fetch_pending_chunks(conn, limit=limit)
    updated = 0

    for batch in batched(pending, batch_size):
        chunk_ids = [chunk_id for chunk_id, _ in batch]
        texts = [text for _, text in batch]
        data = call_embedding_worker(api_url=api_url, texts=texts, batch_size=batch_size)
        embeddings = validate_response(data, expected_count=len(batch), expected_dimension=expected_dimension)
        model_name_or_path = str(data.get("model") or api_url)

        update_embeddings(
            conn,
            rows=list(zip(chunk_ids, embeddings, strict=True)),
            model_name_or_path=model_name_or_path,
        )
        updated += len(batch)
        print(f"embedded {updated}/{len(pending)}")

    return updated


def main() -> None:
    """Запускает расчет embeddings через локальный HTTP worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("EMBEDDING_API_URL", DEFAULT_API_URL))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=1000000, help="Максимум ожидающих чанков")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    args = parser.parse_args()

    with psycopg.connect(database_url()) as conn:
        updated = embed_pending_chunks(
            conn=conn,
            api_url=args.api_url,
            batch_size=args.batch_size,
            limit=args.limit,
            expected_dimension=args.dimension,
        )

    print(f"updated embeddings: {updated}")


if __name__ == "__main__":
    main()
