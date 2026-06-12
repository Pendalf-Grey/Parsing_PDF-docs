from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson


DEFAULT_INPUT = Path("out/parsed_english_text_tables.jsonl")
DEFAULT_OUTPUT = Path("out/chunked_english_text_tables.jsonl")


def normalize_space(text: str) -> str:
    """Нормализует пробелы, не меняя смысловое содержимое текста."""
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Читает JSONL-файл и возвращает каждую строку как словарь."""
    records: list[dict[str, Any]] = []

    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))

    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Записывает словари в JSONL-файл и при необходимости создает родительскую папку."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))


def record_text(record: dict[str, Any], include_tables: bool) -> str:
    """Возвращает текст одной записи Docling, который должен участвовать в чанковании."""
    content_type = record.get("content_type")

    if content_type == "table" and include_tables:
        return normalize_space(record.get("table_markdown") or record.get("text") or "")

    if content_type in {"heading", "text"}:
        return normalize_space(record.get("text") or "")

    return ""


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    """Создает стабильный ключ сортировки для записей из одного входного JSONL."""
    return (
        str(record.get("doc_id") or ""),
        str(record.get("source_file") or ""),
        int(record.get("page_start") or 0),
        int(record.get("_record_index") or 0),
    )


def section_key(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """Группирует записи по документу, исходному файлу и пути секции."""
    return (
        str(record.get("doc_id") or ""),
        str(record.get("source_file") or ""),
        tuple(record.get("section_path") or []),
    )


def words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Разбивает текст на слова и сохраняет символьные смещения каждого слова."""
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def split_text_with_overlap(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Разбивает текст на чанки по словам с overlap между соседними чанками."""
    words = words_with_offsets(text)

    if not words:
        return []

    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_words - overlap_words

    while start < len(words):
        end = min(start + chunk_words, len(words))
        char_start = words[start][1]
        char_end = words[end - 1][2]
        chunks.append(text[char_start:char_end])

        if end == len(words):
            break

        start += step

    return chunks


def chunk_id(doc_id: str, source_file: str, chunk_index: int, text: str) -> str:
    """Создает стабильный chunk_id из метаданных, порядка чанка и текста."""
    payload = f"{doc_id}\n{source_file}\n{chunk_index}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def page_range(records: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Возвращает минимальную начальную и максимальную конечную страницу исходных записей."""
    starts = [r.get("page_start") for r in records if r.get("page_start") is not None]
    ends = [r.get("page_end") for r in records if r.get("page_end") is not None]

    if not starts or not ends:
        return None, None

    return min(starts), max(ends)


def build_chunk_record(
    source_records: list[dict[str, Any]],
    text: str,
    chunk_index: int,
    section_path: list[str],
) -> dict[str, Any]:
    """Создает одну JSONL-ready запись чанка и сохраняет исходные метаданные."""
    first = source_records[0]
    page_start, page_end = page_range(source_records)
    doc_id = str(first.get("doc_id") or "")
    source_file = str(first.get("source_file") or "")

    return {
        "chunk_id": chunk_id(doc_id, source_file, chunk_index, text),
        "doc_id": doc_id,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": section_path,
        "text": text,
        "word_count": len(words_with_offsets(text)),
        "source_record_indices": [r.get("_record_index") for r in source_records],
        "source_content_types": sorted({str(r.get("content_type") or "") for r in source_records}),
    }


def flush_group(
    group_records: list[dict[str, Any]],
    chunk_words: int,
    overlap_words: int,
    chunk_index_start: int,
    include_tables: bool,
) -> list[dict[str, Any]]:
    """Преобразует группу записей одной секции Docling в чанки с overlap."""
    if not group_records:
        return []

    section_path = list(group_records[0].get("section_path") or [])
    combined_parts = [record_text(r, include_tables=include_tables) for r in group_records]
    combined_text = normalize_space("\n\n".join(part for part in combined_parts if part))

    if not combined_text:
        return []

    chunk_texts = split_text_with_overlap(combined_text, chunk_words, overlap_words)
    chunks: list[dict[str, Any]] = []

    for offset, chunk_text in enumerate(chunk_texts):
        chunks.append(
            build_chunk_record(
                source_records=group_records,
                text=chunk_text,
                chunk_index=chunk_index_start + offset,
                section_path=section_path,
            )
        )

    return chunks


def validate_chunk_settings(chunk_words: int, overlap_words: int) -> None:
    """Проверяет размер чанка и overlap перед обработкой."""
    if chunk_words <= 0:
        raise ValueError("--chunk-words must be greater than 0")

    if overlap_words < 0:
        raise ValueError("--overlap-words cannot be negative")

    if overlap_words >= chunk_words:
        raise ValueError("--overlap-words must be smaller than --chunk-words")


def chunk_records(
    records: list[dict[str, Any]],
    chunk_words: int,
    overlap_words: int,
    include_tables: bool,
) -> list[dict[str, Any]]:
    """Чанкует записи Docling по секциям документа с настраиваемым overlap."""
    validate_chunk_settings(chunk_words, overlap_words)

    prepared_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prepared = dict(record)
        prepared["_record_index"] = index

        if record_text(prepared, include_tables=include_tables):
            prepared_records.append(prepared)

    prepared_records.sort(key=record_sort_key)

    chunks: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = []
    current_key: tuple[str, str, tuple[str, ...]] | None = None

    for record in prepared_records:
        key = section_key(record)

        if current_key is not None and key != current_key:
            chunks.extend(flush_group(current_group, chunk_words, overlap_words, len(chunks), include_tables))
            current_group = []

        current_key = key
        current_group.append(record)

    chunks.extend(flush_group(current_group, chunk_words, overlap_words, len(chunks), include_tables))

    return chunks


def main() -> None:
    """Запускает CLI для чанкования JSONL, полученного из Docling."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Входной JSONL из parce_docling_english.py")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Путь для выходного JSONL с чанками")
    parser.add_argument("--chunk-words", type=int, default=180, help="Максимальное количество слов в чанке")
    parser.add_argument("--overlap-words", type=int, default=40, help="Слова, повторяемые из предыдущего чанка")
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Исключить табличные записи из чанкования",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    chunks = chunk_records(
        records=records,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        include_tables=not args.no_tables,
    )
    write_jsonl(Path(args.out), chunks)

    print(f"loaded records: {len(records)}")
    print(f"saved chunks: {len(chunks)}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
