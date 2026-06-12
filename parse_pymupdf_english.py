from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import fitz
import orjson
from lingua import Language, LanguageDetectorBuilder


ENGLISH_DETECTOR = (
    LanguageDetectorBuilder
    .from_languages(
        Language.ENGLISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.ITALIAN,
        Language.SPANISH,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.CHINESE,
        Language.JAPANESE,
        Language.KOREAN,
    )
    .with_preloaded_language_models()
    .build()
)


TECHNICAL_ENGLISH_HINTS = {
    "server", "power", "supply", "drive", "system", "board", "processor",
    "memory", "rack", "remove", "replace", "component", "cable", "fan",
    "heatsink", "battery", "diagnostic", "warning", "caution", "note",
    "customer", "repair", "mandatory", "optional", "description",
    "spare", "part", "number", "item",
}


NON_ENGLISH_STRONG_MARKERS = {
    "obligatoire", "facultatif", "remarque", "réparation",
    "obbligatorie", "opzionali", "riparazione",
    "zwingend", "hinweis", "austausch",
    "obligatorio", "opcional", "reparaciones",
    "verplicht", "optioneel", "onderdelen",
    "obrigatória", "opcional", "observação",
}


def sha256_file(path: Path) -> str:
    """Считает SHA-256 для исходного PDF."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """Нормализует пробелы и приводит текст к нижнему регистру."""
    text = text.replace("\u00ad", "")
    text = text.replace("￾", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().lower()


def has_cjk(text: str) -> bool:
    """Возвращает true, если текст содержит китайские, японские или корейские символы."""
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))


def latin_letter_ratio(text: str) -> float:
    """Возвращает долю ASCII-латиницы среди всех латинских букв."""
    letters = re.findall(r"[A-Za-zÀ-ÿ]", text)
    if not letters:
        return 0.0
    ascii_letters = re.findall(r"[A-Za-z]", text)
    return len(ascii_letters) / max(len(letters), 1)


def looks_like_technical_english(text: str) -> bool:
    """Определяет короткие технические английские строки по доменным подсказкам."""
    words = set(re.findall(r"[A-Za-z]{3,}", text.lower()))
    return len(words & TECHNICAL_ENGLISH_HINTS) >= 2


def has_non_english_markers(text: str) -> bool:
    """Ищет сильные маркеры неанглийских warranty/service-блоков."""
    low = text.lower()
    return any(marker in low for marker in NON_ENGLISH_STRONG_MARKERS)


def is_english(text: str, min_confidence: float = 0.60) -> bool:
    """Определяет, достаточно ли текстовый блок похож на английский для этого технического корпуса."""
    text = normalize_text(text)

    if len(text) < 8:
        return False

    if has_cjk(text):
        return False

    if has_non_english_markers(text) and not looks_like_technical_english(text):
        return False

    if looks_like_technical_english(text) and latin_letter_ratio(text) > 0.85:
        return True

    if len(text) >= 40:
        language = ENGLISH_DETECTOR.detect_language_of(text)
        confidence = ENGLISH_DETECTOR.compute_language_confidence(text, Language.ENGLISH)
        return language == Language.ENGLISH and confidence >= min_confidence

    return latin_letter_ratio(text) > 0.95 and looks_like_technical_english(text)


def block_text(block: dict[str, Any]) -> str:
    """Извлекает нормализованный текст из текстового блока PyMuPDF."""
    lines: list[str] = []

    for line in block.get("lines", []):
        spans = line.get("spans", [])
        text = "".join(str(span.get("text", "")) for span in spans)
        text = normalize_text(text)

        if text:
            lines.append(text)

    return normalize_text("\n".join(lines))


def block_max_font_size(block: dict[str, Any]) -> float:
    """Возвращает максимальный размер шрифта в текстовом блоке PyMuPDF."""
    sizes: list[float] = []

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            size = span.get("size")
            if size is not None:
                sizes.append(float(size))

    return max(sizes) if sizes else 0.0


def page_body_font_size(blocks: list[dict[str, Any]]) -> float:
    """Оценивает основной размер шрифта страницы по текстовым spans."""
    sizes: list[float] = []

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = normalize_text(str(span.get("text", "")))
                size = span.get("size")
                if text and size is not None:
                    sizes.append(round(float(size), 1))

    if not sizes:
        return 0.0

    counts: dict[float, int] = {}
    for size in sizes:
        counts[size] = counts.get(size, 0) + 1

    return max(counts, key=counts.get)


def looks_like_heading(text: str, font_size: float, body_size: float) -> bool:
    """Эвристически определяет, является ли текстовый блок заголовком."""
    words = text.split()

    if len(words) > 12:
        return False

    if text.endswith("."):
        return False

    if body_size and font_size >= body_size + 1.5:
        return True

    return len(words) <= 6 and looks_like_technical_english(text)


def extract_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """Возвращает текстовые блоки, отсортированные сверху вниз и слева направо."""
    data = page.get_text("dict")
    blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]
    return sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))


def parse_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Парсит PDF через PyMuPDF и возвращает JSONL-ready записи английского текста."""
    doc_id = sha256_file(pdf_path)
    records: list[dict[str, Any]] = []
    current_section: str | None = None

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            blocks = extract_blocks(page)
            body_size = page_body_font_size(blocks)

            for block in blocks:
                text = block_text(block)

                if not text or not is_english(text):
                    continue

                max_size = block_max_font_size(block)
                is_heading = looks_like_heading(text, max_size, body_size)

                if is_heading:
                    current_section = text

                records.append({
                    "doc_id": doc_id,
                    "source_file": pdf_path.name,
                    "content_type": "heading" if is_heading else "text",
                    "page_start": page_index,
                    "page_end": page_index,
                    "section_path": [current_section] if current_section else [],
                    "text": text,
                    "table_markdown": None,
                    "table_json": None,
                })

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Записывает записи в JSONL."""
    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))


def write_debug_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    """Записывает человекочитаемый debug Markdown-файл."""
    chunks: list[str] = []

    for i, record in enumerate(records, start=1):
        page = (
            str(record["page_start"])
            if record["page_start"] == record["page_end"]
            else f'{record["page_start"]}-{record["page_end"]}'
        )

        chunks.append(f'\n\n---\n\n### {i}. {record["content_type"]} | page {page}\n')

        if record["section_path"]:
            chunks.append("**Section:** " + " > ".join(record["section_path"]) + "\n\n")

        chunks.append(record["text"])

    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    """Запускает CLI парсера PyMuPDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PDF-файл или папка с PDF")
    parser.add_argument("--out", required=True, help="Выходная папка")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = sorted(input_path.glob("*.pdf"))

    all_records: list[dict[str, Any]] = []

    for pdf in pdfs:
        print(f"Parsing with PyMuPDF: {pdf}")
        records = parse_pdf(pdf)
        print(f"  extracted English records: {len(records)}")
        all_records.extend(records)

    jsonl_path = out_dir / "parsed_pymupdf_english_text.jsonl"
    debug_md_path = out_dir / "parsed_pymupdf_english_text.debug.md"

    write_jsonl(jsonl_path, all_records)
    write_debug_markdown(debug_md_path, all_records)

    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Saved debug Markdown: {debug_md_path}")


if __name__ == "__main__":
    main()
