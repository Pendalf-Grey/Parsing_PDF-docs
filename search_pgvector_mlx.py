from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from pathlib import Path

import psycopg

from embed_chunks_pgvector_mlx import DEFAULT_DIMENSION, DEFAULT_MAX_LENGTH, DEFAULT_MODEL, MlxQwen3Embedder


def database_url() -> str:
    """Возвращает URL подключения к PostgreSQL из окружения."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError("DATABASE_URL is required, for example postgresql://user:pass@localhost:5432/db")

    return url


def vector_literal(vector: Sequence[float]) -> str:
    """Сериализует Python-последовательность float в формат литерала pgvector."""
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Возвращает L2-нормализованную копию embedding-вектора."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))

    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding vector")

    return [float(value) / norm for value in vector]


def search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    limit: int,
    candidate_limit: int,
) -> list[tuple]:
    """Запускает ANN-поиск по полному 1024-мерному embedding-вектору."""
    query_vector = vector_literal(query_embedding)

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
                SELECT
                    chunk_id,
                    source_file,
                    page_start,
                    page_end,
                    section_path,
                    text,
                    embedding
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            )
            SELECT
                chunk_id,
                source_file,
                page_start,
                page_end,
                section_path,
                left(text, 500) AS preview,
                embedding <=> %s::vector AS distance
            FROM candidates
            ORDER BY distance
            LIMIT %s
            """,
            (query_vector, candidate_limit, query_vector, limit),
        )
        return cur.fetchall()


def main() -> None:
    """Считает embedding запроса через MLX Qwen3 и ищет ближайшие чанки в pgvector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Текст поискового запроса")
    parser.add_argument("--model", default=os.getenv("QWEN3_EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--no-normalize", action="store_true", help="Использовать сырой вектор запроса")
    args = parser.parse_args()

    selected_model = args.model

    if Path(selected_model).exists():
        selected_model = str(Path(selected_model))

    embedder = MlxQwen3Embedder(selected_model, max_length=args.max_length)
    embedding = embedder.embed([args.query])[0]

    if len(embedding) != args.dimension:
        raise ValueError(f"Expected query embedding dimension {args.dimension}, got {len(embedding)}")

    query_embedding = embedding if args.no_normalize else normalize_vector(embedding)

    with psycopg.connect(database_url()) as conn:
        rows = search(
            conn,
            query_embedding=query_embedding,
            limit=args.limit,
            candidate_limit=args.candidate_limit,
        )

    for rank, row in enumerate(rows, start=1):
        chunk_id, source_file, page_start, page_end, section_path, preview, distance = row
        print(f"\n#{rank} distance={distance:.6f} pages={page_start}-{page_end}")
        print(f"chunk_id: {chunk_id}")
        print(f"source_file: {source_file}")
        print(f"section_path: {' > '.join(section_path)}")
        print(preview)


if __name__ == "__main__":
    main()
