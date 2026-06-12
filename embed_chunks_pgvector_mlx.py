from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psycopg


DEFAULT_MODEL = "models/Qwen3-Embedding-0.6B"
DEFAULT_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 8192


def database_url() -> str:
    """Возвращает URL подключения к PostgreSQL из окружения."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError("DATABASE_URL is required, for example postgresql://user:pass@localhost:5432/db")

    return url


def model_path(default: str) -> str:
    """Возвращает путь к embedding-модели из CLI-значения по умолчанию или окружения."""
    return os.getenv("QWEN3_EMBEDDING_MODEL", default)


def fetch_pending_chunks(conn: psycopg.Connection, limit: int) -> list[tuple[str, str]]:
    """Получает чанки, у которых еще нет embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, text
            FROM document_chunks
            WHERE embedding IS NULL
            ORDER BY chunk_index
            LIMIT %s
            """,
            (limit,),
        )
        return [(str(chunk_id), str(text)) for chunk_id, text in cur.fetchall()]


def vector_literal(vector: Sequence[float]) -> str:
    """Сериализует Python-последовательность float в формат литерала pgvector."""
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Возвращает L2-нормализованную копию embedding-вектора."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))

    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding vector")

    return [float(value) / norm for value in vector]


def batched(items: Sequence[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    """Разбивает элементы на батчи фиксированного размера."""
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


class MlxQwen3Embedder:
    """Тонкая обертка над mlx-embeddings для текстовых embeddings Qwen3."""

    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        from mlx_embeddings.utils import load

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.model, self.tokenizer = load(model_name_or_path)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Считает embeddings для батча текстов и возвращает Python-векторы float."""
        import mlx.core as mx

        inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        embeddings = outputs.text_embeds
        mx.eval(embeddings)
        return embeddings.tolist()


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


def validate_embedding(vector: Sequence[float], expected_dimension: int) -> None:
    """Проверяет, что embedding имеет ожидаемую размерность pgvector."""
    if len(vector) != expected_dimension:
        raise ValueError(f"Expected embedding dimension {expected_dimension}, got {len(vector)}")


def embed_pending_chunks(
    conn: psycopg.Connection,
    embedder: MlxQwen3Embedder,
    batch_size: int,
    limit: int,
    expected_dimension: int,
    normalize: bool,
) -> int:
    """Считает embeddings для ожидающих чанков и возвращает число обновленных строк."""
    pending = fetch_pending_chunks(conn, limit=limit)
    updated = 0

    for batch in batched(pending, batch_size):
        chunk_ids = [chunk_id for chunk_id, _ in batch]
        texts = [text for _, text in batch]
        embeddings = embedder.embed(texts)
        prepared: list[tuple[str, list[float]]] = []

        for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
            validate_embedding(embedding, expected_dimension)
            prepared.append((chunk_id, normalize_vector(embedding) if normalize else embedding))

        update_embeddings(conn, prepared, model_name_or_path=embedder.model_name_or_path)
        updated += len(prepared)
        print(f"embedded {updated}/{len(pending)}")

    return updated


def main() -> None:
    """Считает embeddings для PostgreSQL-чанков через MLX Qwen3 и сохраняет их в pgvector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Локальный путь или id модели Hugging Face")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=1000000, help="Максимум ожидающих чанков для embedding")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--no-normalize", action="store_true", help="Сохранять сырые векторы модели")
    args = parser.parse_args()

    selected_model = model_path(args.model)

    if Path(selected_model).exists():
        selected_model = str(Path(selected_model))

    embedder = MlxQwen3Embedder(selected_model, max_length=args.max_length)

    with psycopg.connect(database_url()) as conn:
        updated = embed_pending_chunks(
            conn=conn,
            embedder=embedder,
            batch_size=args.batch_size,
            limit=args.limit,
            expected_dimension=args.dimension,
            normalize=not args.no_normalize,
        )

    print(f"updated embeddings: {updated}")


if __name__ == "__main__":
    main()
