from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from lingua import Language, LanguageDetectorBuilder


def patch_pydantic_string_constraints() -> None:
    """Чинит конструктор StringConstraints в связке Python 3.10 + pydantic/Docling."""
    try:
        from pydantic import StringConstraints
    except Exception:
        return

    try:
        StringConstraints(strict=True, pattern="x")
        return
    except TypeError:
        pass

    def init(
        self,
        strip_whitespace=None,
        to_upper=None,
        to_lower=None,
        strict=None,
        min_length=None,
        max_length=None,
        pattern=None,
    ):
        object.__setattr__(self, "strip_whitespace", strip_whitespace)
        object.__setattr__(self, "to_upper", to_upper)
        object.__setattr__(self, "to_lower", to_lower)
        object.__setattr__(self, "strict", strict)
        object.__setattr__(self, "min_length", min_length)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "pattern", pattern)

    StringConstraints.__init__ = init


patch_pydantic_string_constraints()

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

try:
    from docling.datamodel.pipeline_options import TableFormerMode
except Exception:
    TableFormerMode = None


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
    # Частые слова из warranty-блоков на FR/DE/IT/ES/PT/NL
    "obligatoire", "facultatif", "remarque", "réparation",
    "obbligatorie", "opzionali", "riparazione",
    "zwingend", "hinweis", "austausch",
    "obligatorio", "opcional", "reparaciones",
    "verplicht", "optioneel", "onderdelen",
    "obrigatória", "opcional", "observação",
}


def sha256_file(path: Path) -> str:
    """Считает SHA-256 хеш файла, чтобы использовать его как стабильный doc_id."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """Очищает текст от служебных символов, нормализует пробелы и приводит к нижнему регистру."""
    text = text.replace("\u00ad", "")  # Мягкий перенос.
    text = text.replace("￾", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().lower()


def check_device_available(device: str) -> None:
    """Проверяет доступность выбранного ускорителя и предупреждает о вероятном fallback на CPU."""
    if device != "mps":
        return

    try:
        import torch
    except Exception:
        warnings.warn(
            "запрошен --device mps, но torch не импортируется; docling может откатиться на cpu",
            stacklevel=2,
        )
        return

    if not torch.backends.mps.is_available():
        warnings.warn(
            "запрошен --device mps, но pytorch не сообщает о доступности apple gpu/mps; "
            "docling может откатиться на cpu",
            stacklevel=2,
        )


def has_cjk(text: str) -> bool:
    """Определяет, содержит ли текст китайские, японские или корейские символы."""
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))


def latin_letter_ratio(text: str) -> float:
    """Возвращает долю ASCII-латиницы среди всех латинских букв в тексте."""
    letters = re.findall(r"[A-Za-zÀ-ÿ]", text)
    if not letters:
        return 0.0
    ascii_letters = re.findall(r"[A-Za-z]", text)
    return len(ascii_letters) / max(len(letters), 1)


def looks_like_technical_english(text: str) -> bool:
    """Проверяет наличие технических англоязычных слов, типичных для service guide."""
    words = set(re.findall(r"[A-Za-z]{3,}", text.lower()))
    return len(words & TECHNICAL_ENGLISH_HINTS) >= 2


def has_non_english_markers(text: str) -> bool:
    """Ищет сильные маркеры неанглийских warranty/repair-блоков."""
    low = text.lower()
    return any(marker in low for marker in NON_ENGLISH_STRONG_MARKERS)


def is_english(text: str, min_confidence: float = 0.60) -> bool:
    """Определяет, можно ли считать фрагмент английским с учетом технических эвристик."""
    text = normalize_text(text)

    if len(text) < 8:
        return False

    if has_cjk(text):
        return False

    if has_non_english_markers(text) and not looks_like_technical_english(text):
        return False

    # Для артикулов, part numbers и коротких технических строк.
    if looks_like_technical_english(text) and latin_letter_ratio(text) > 0.85:
        return True

    # Для длинных обычных абзацев.
    if len(text) >= 40:
        language = ENGLISH_DETECTOR.detect_language_of(text)
        confidence = ENGLISH_DETECTOR.compute_language_confidence(text, Language.ENGLISH)
        return language == Language.ENGLISH and confidence >= min_confidence

    # Для коротких заголовков.
    return latin_letter_ratio(text) > 0.95 and looks_like_technical_english(text)


def get_page_range(item: Any) -> tuple[int | None, int | None]:
    """Извлекает начальную и конечную страницу элемента Docling из provenance-метаданных."""
    prov = getattr(item, "prov", None) or []
    pages = [p.page_no for p in prov if getattr(p, "page_no", None) is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def get_label(item: Any) -> str:
    """Возвращает label элемента Docling в нижнем регистре для удобной классификации."""
    label = getattr(item, "label", "")
    return str(label).lower()


def get_item_text(item: Any) -> str:
    """Достает и нормализует текст элемента Docling из поля text или orig."""
    text = getattr(item, "text", None)
    if text:
        return normalize_text(str(text))

    orig = getattr(item, "orig", None)
    if orig:
        return normalize_text(str(orig))

    return ""


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, str]]:
    """Преобразует таблицу pandas в список словарей с нормализованными строковыми значениями."""
    df = df.fillna("")
    df.columns = [normalize_text(str(c)) for c in df.columns]
    records: list[dict[str, str]] = []

    for row in df.to_dict(orient="records"):
        records.append({
            normalize_text(str(k)): normalize_text(str(v))
            for k, v in row.items()
        })

    return records


def table_to_markdown(df: pd.DataFrame) -> str:
    """Преобразует таблицу pandas в Markdown-представление без индекса."""
    df = df.fillna("")
    df.columns = [normalize_text(str(c)) for c in df.columns]
    df = df.map(lambda value: normalize_text(str(value)))
    return df.to_markdown(index=False)


def table_plain_text(df: pd.DataFrame) -> str:
    """Собирает заголовки и ячейки таблицы в одну строку для фильтрации и поиска."""
    parts: list[str] = []

    for col in df.columns:
        parts.append(str(col))

    for row in df.fillna("").astype(str).values.tolist():
        parts.extend(row)

    return normalize_text(" ".join(parts))


def is_english_table(df: pd.DataFrame) -> bool:
    """Определяет, является ли таблица англоязычной или важной parts-таблицей HP."""
    flat = table_plain_text(df)

    if not flat:
        return False

    # В HP service guides важные таблицы почти всегда имеют такие заголовки.
    header_text = " ".join(map(str, df.columns)).lower()
    if (
        "item" in header_text
        and "description" in header_text
        and ("spare" in header_text or "part" in header_text)
    ):
        return True

    return is_english(flat, min_confidence=0.50)


def build_converter(do_ocr: bool, device: str) -> DocumentConverter:
    """Создает DocumentConverter с настройками OCR, таблиц и аппаратного ускорителя."""
    pipeline_options = PdfPipelineOptions()

    if hasattr(pipeline_options, "accelerator_options"):
        pipeline_options.accelerator_options.device = device

    if hasattr(pipeline_options, "do_ocr"):
        pipeline_options.do_ocr = do_ocr

    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = True

    if hasattr(pipeline_options, "table_structure_options"):
        pipeline_options.table_structure_options.do_cell_matching = True

        if TableFormerMode is not None and hasattr(pipeline_options.table_structure_options, "mode"):
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # OCR только на английском. Для digital PDF можно оставить --ocr выключенным.
    if do_ocr and hasattr(pipeline_options, "ocr_options"):
        try:
            pipeline_options.ocr_options.lang = ["en"]
        except Exception:
            pass

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def parse_pdf(pdf_path: Path, converter: DocumentConverter) -> list[dict[str, Any]]:
    """Парсит PDF и возвращает JSONL-ready записи по английскому тексту, заголовкам и таблицам."""
    doc_hash = sha256_file(pdf_path)
    result = converter.convert(pdf_path)
    doc = result.document

    records: list[dict[str, Any]] = []

    current_title: str | None = None
    section_stack: dict[int, str] = {}

    for item, level in doc.iterate_items():
        label = get_label(item)
        page_start, page_end = get_page_range(item)

        # Пропускаем колонтитулы и картинки.
        if "page_header" in label or "page_footer" in label or "picture" in label:
            continue

        # Заголовки.
        if "title" in label or "section_header" in label:
            text = get_item_text(item)
            if is_english(text):
                if "title" in label:
                    current_title = text

                section_stack[level] = text
                for k in list(section_stack.keys()):
                    if k > level:
                        del section_stack[k]

                records.append({
                    "doc_id": doc_hash,
                    "source_file": pdf_path.name,
                    "content_type": "heading",
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": [section_stack[k] for k in sorted(section_stack)],
                    "text": text,
                    "table_markdown": None,
                    "table_json": None,
                })
            continue

        # Таблицы.
        if "table" in label and hasattr(item, "export_to_dataframe"):
            try:
                df = item.export_to_dataframe(doc=doc)
            except TypeError:
                df = item.export_to_dataframe()
            except Exception:
                continue

            if df is None or df.empty:
                continue

            if not is_english_table(df):
                continue

            table_md = table_to_markdown(df)
            table_text = table_plain_text(df)

            records.append({
                "doc_id": doc_hash,
                "source_file": pdf_path.name,
                "content_type": "table",
                "page_start": page_start,
                "page_end": page_end,
                "section_path": [section_stack[k] for k in sorted(section_stack)],
                "text": table_text,
                "table_markdown": table_md,
                "table_json": dataframe_to_records(df),
            })
            continue

        # Обычный текст / списки.
        text = get_item_text(item)
        if not text:
            continue

        if not is_english(text):
            continue

        records.append({
            "doc_id": doc_hash,
            "source_file": pdf_path.name,
            "content_type": "text",
            "page_start": page_start,
            "page_end": page_end,
            "section_path": [section_stack[k] for k in sorted(section_stack)],
            "text": text,
            "table_markdown": None,
            "table_json": None,
        })

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Записывает список records в JSONL-файл через быстрый orjson."""
    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))


def write_debug_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    """Создает человекочитаемый Markdown-отчет для ручной проверки извлеченных records."""
    chunks: list[str] = []

    for i, r in enumerate(records, start=1):
        page = (
            str(r["page_start"])
            if r["page_start"] == r["page_end"]
            else f'{r["page_start"]}-{r["page_end"]}'
        )

        chunks.append(f'\n\n---\n\n### {i}. {r["content_type"]} | page {page}\n')

        if r["section_path"]:
            chunks.append("**Section:** " + " > ".join(r["section_path"]) + "\n\n")

        if r["content_type"] == "table":
            chunks.append(r["table_markdown"] or "")
        else:
            chunks.append(r["text"])

    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    """Разбирает CLI-аргументы, запускает парсинг PDF и сохраняет JSONL/debug Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PDF-файл или папка с PDF")
    parser.add_argument("--out", required=True, help="Выходная папка")
    parser.add_argument("--ocr", action="store_true", help="Включить OCR для сканированных PDF")
    parser.add_argument(
        "--device",
        default="auto",
        help="Устройство-ускоритель Docling: auto, cpu, mps, cuda, cuda:N или xpu",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = sorted(input_path.glob("*.pdf"))

    check_device_available(args.device)
    converter = build_converter(do_ocr=args.ocr, device=args.device)

    all_records: list[dict[str, Any]] = []

    for pdf in pdfs:
        print(f"Parsing: {pdf}")
        records = parse_pdf(pdf, converter)
        print(f"  extracted English records: {len(records)}")
        all_records.extend(records)

    jsonl_path = out_dir / "parsed_english_text_tables.jsonl"
    debug_md_path = out_dir / "parsed_english_text_tables.debug.md"

    write_jsonl(jsonl_path, all_records)
    write_debug_markdown(debug_md_path, all_records)

    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Saved debug Markdown: {debug_md_path}")


if __name__ == "__main__":
    main()
