import re
import shutil
import time
import unicodedata
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

POSITIVE_WORDS = {
    "excelente", "perfecto", "increible", "maravilloso", "fantastico",
    "genial", "buenisimo", "recomendable", "bueno",
}

POSITIVE_PHRASES = {
    "muy bueno", "muy buena", "super bien", "vale la pena",
}

NEGATIVE_WORDS = {
    "lento", "lenta", "calienta", "calentamiento", "bateria", "pila",
    "pantalla", "teclado", "falla", "fallo", "ruidoso", "traba",
    "congela", "defecto", "problema", "malo", "mala",
}

NEGATIVE_PHRASES = {
    "no sirve", "no enciende", "no carga", "se traba", "se congela",
    "no funciona", "muy lento", "muy lenta",
}

CONTRAST_REGEX = re.compile(
    r"\b(pero|aunque|sin\s+embargo|a\s+pesar\s+de|no\s+obstante)\b",
    flags=re.IGNORECASE,
)

SARCASM_CUE_REGEX = re.compile(
    r"\b(jajaja|jaja|jeje|ja|ajaja|uff|aja|sarcasmo|claro|"
    r"si\s+claro|como\s+no)\b|/s",
    flags=re.IGNORECASE,
)


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path} con las codificaciones probadas.")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if text.endswith(".") and not text.endswith("..."):
        text = text[:-1].rstrip()
    return text


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def normalize_for_match(text: str) -> str:
    return strip_accents(text.lower())


def tokenize_words(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    return set(re.findall(r"[a-z]+", normalized))


def has_keywords(text: str, words: set[str], phrases: set[str]) -> bool:
    normalized = normalize_for_match(text)
    if tokenize_words(normalized).intersection(words):
        return True
    return any(phrase in normalized for phrase in phrases)


def detect_sarcasm(text: str) -> tuple[bool, str]:
    normalized = normalize_for_match(text)
    has_positive = has_keywords(normalized, POSITIVE_WORDS, POSITIVE_PHRASES)
    has_negative = has_keywords(normalized, NEGATIVE_WORDS, NEGATIVE_PHRASES)
    has_contrast = bool(CONTRAST_REGEX.search(normalized))
    has_cue = bool(SARCASM_CUE_REGEX.search(normalized))

    reasons = []
    if has_cue:
        reasons.append("cue")
    if has_positive and has_negative and has_contrast:
        reasons.append("positivo_vs_negativo")

    if reasons:
        return True, "+".join(reasons)
    return False, ""


def remove_sarcasm_cues(text: str) -> str:
    return SARCASM_CUE_REGEX.sub("", text)


def split_on_contrast(text: str) -> tuple[str, str]:
    match = CONTRAST_REGEX.search(text)
    if not match:
        return text, ""
    before = text[:match.start()].strip(" ,.;:-")
    after = text[match.end():].strip(" ,.;:-")
    return before, after


def clean_sarcasm_text(text: str) -> tuple[str, str]:
    cleaned = remove_sarcasm_cues(text)
    cleaned = normalize_text(cleaned)

    before, after = split_on_contrast(cleaned)
    if not after:
        return cleaned, "cues_removed"

    before_negative = has_keywords(before, NEGATIVE_WORDS, NEGATIVE_PHRASES)
    after_negative = has_keywords(after, NEGATIVE_WORDS, NEGATIVE_PHRASES)
    before_positive = has_keywords(before, POSITIVE_WORDS, POSITIVE_PHRASES)
    after_positive = has_keywords(after, POSITIVE_WORDS, POSITIVE_PHRASES)

    if after_negative and (before_positive or not before_negative):
        return normalize_text(after), "kept_after_contrast"
    if before_negative and after_positive:
        return normalize_text(before), "kept_before_contrast"

    return cleaned, "contrast_kept_full"


def main() -> None:
    df = load_csv(CSV_PATH)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna final_clean_text.")

    total_rows = len(df)
    rows_processed = 0
    rows_skipped = 0
    sarcasm_detected = 0
    sarcasm_cleaned = 0
    changes_log = []

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Procesando", unit="fila"):
        text = row.get("final_clean_text")
        if not isinstance(text, str) or not text.strip():
            rows_skipped += 1
            continue

        rows_processed += 1
        original = text
        text = normalize_text(text)

        detected, _ = detect_sarcasm(text)
        if detected:
            sarcasm_detected += 1
            cleaned, _ = clean_sarcasm_text(text)
            if cleaned and cleaned != original:
                text = cleaned
                sarcasm_cleaned += 1
                review_id = row.get("id_review") or f"row_{idx}"
                changes_log.append((review_id, original, text))

        if text != original:
            df.at[idx, "final_clean_text"] = text

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".bak.{timestamp}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    report_path = BASE_DIR / f"sarcasmo_casos_corregidos_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Casos de sarcasmo corregidos (antes y despues)\n")
        f.write("=" * 72 + "\n")
        for review_id, before, after in changes_log:
            f.write(f"ID: {review_id}\n")
            f.write(f"ANTES: {before}\n")
            f.write(f"DESPUES: {after}\n")
            f.write("-" * 72 + "\n")

    print("--- REPORTE DE SARCASMO ---")
    print(f"Filas procesadas: {rows_processed}")
    print(f"Filas omitidas (sin texto): {rows_skipped}")
    print(f"Filas con sarcasmo detectado: {sarcasm_detected}")
    print(f"Filas con sarcasmo limpiado: {sarcasm_cleaned}")
    print(f"Backup creado: {backup_path}")
    print(f"Reporte de casos corregidos: {report_path}")


if __name__ == "__main__":
    main()
