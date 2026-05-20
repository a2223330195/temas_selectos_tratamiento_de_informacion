import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

TAG_REGEX = re.compile(
    r"(?<!\w)/s(?!\w)|\b(sarcasmo|sarcastic[oa]?|ironia|ironico|ironica)\b",
    flags=re.IGNORECASE,
)
CUE_REGEX = re.compile(
    r"\b(si\s+claro|claro\s+que\s+si|como\s+no|aja|aja\s+si|seguro\s+que\s+si)\b",
    flags=re.IGNORECASE,
)
LAUGHTER_REGEX = re.compile(
    r"\b(jajaja|jaja|jeje|jiji|ja)\b",
    flags=re.IGNORECASE,
)
CONTRAST_REGEX = re.compile(
    r"\b(pero|aunque|sin\s+embargo|a\s+pesar\s+de|no\s+obstante)\b",
    flags=re.IGNORECASE,
)

POSITIVE_WORDS = {
    "excelente", "perfecto", "increible", "maravilloso", "fantastico",
    "genial", "buenisimo", "recomendable", "buen", "bueno", "buena",
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

TECH_NEGATIVE_WORDS = {
    "bateria", "pila", "pantalla", "teclado", "calienta", "calentamiento",
    "ruidoso", "ventilador", "temperatura",
}

SARCASM_PHRASE_REGEX = re.compile(
    r"(pisapapeles|tirar el dinero|para tirar el dinero|freir un huevo|"
    r"gracias por nada|para la basura|de adorno|excelente para tirar)",
    flags=re.IGNORECASE,
)

QUOTE_POSITIVE_REGEX = re.compile(
    r"[\"\'“”](excelente|perfecto|increible|genial|buenisimo|recomendable)[\"\'“”]",
    flags=re.IGNORECASE,
)

MIN_SCORE = 4

WORD_PATTERN = re.compile(r"[a-z]+", flags=re.IGNORECASE)


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path} con las codificaciones probadas.")


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def normalize_for_match(text: str) -> str:
    return strip_accents(text.lower())


def tokenize_words(text: str) -> set[str]:
    return set(WORD_PATTERN.findall(text))


def has_keywords(text: str, words: set[str], phrases: set[str]) -> bool:
    if tokenize_words(text).intersection(words):
        return True
    return any(phrase in text for phrase in phrases)


def has_negative_non_technical(text: str) -> bool:
    tokens = tokenize_words(text)
    negative_hits = tokens.intersection(NEGATIVE_WORDS)
    if not negative_hits:
        return False
    return bool(negative_hits - TECH_NEGATIVE_WORDS)


def detect_sarcasm_candidate(text: str) -> tuple[int, str]:
    normalized = normalize_for_match(text)
    reasons = []
    score = 0

    if TAG_REGEX.search(normalized):
        score += 4
        reasons.append("tag")

    if SARCASM_PHRASE_REGEX.search(normalized):
        score += 4
        reasons.append("frase_sarcastica")

    has_cue = bool(CUE_REGEX.search(normalized))
    has_positive = has_keywords(normalized, POSITIVE_WORDS, POSITIVE_PHRASES)
    has_negative = has_negative_non_technical(normalized)
    has_contrast = bool(CONTRAST_REGEX.search(normalized))
    has_quote_positive = bool(QUOTE_POSITIVE_REGEX.search(normalized))

    if has_cue and has_negative:
        score += 2
        reasons.append("cue")
        if has_positive:
            score += 1
            reasons.append("positivo")
        if has_contrast:
            score += 1
            reasons.append("contraste")
        if has_quote_positive:
            score += 1
            reasons.append("comillas_positivas")

    return score, "+".join(reasons)


def main() -> None:
    df = load_csv(CSV_PATH)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna final_clean_text.")

    total_rows = len(df)
    rows_skipped = 0
    sarcasm_detected = 0
    results = []

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Analizando", unit="fila"):
        text = row.get("final_clean_text")
        if not isinstance(text, str) or not text.strip():
            rows_skipped += 1
            continue

        score, reason = detect_sarcasm_candidate(text)
        if score >= MIN_SCORE:
            sarcasm_detected += 1
            review_id = row.get("id_review") or f"row_{idx}"
            rating_num = row.get("rating_num")
            rating = row.get("rating")
            results.append((score, review_id, rating_num, rating, reason, text))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = BASE_DIR / f"sarcasmo_casos_posibles_{timestamp}.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Casos posibles de sarcasmo (revision manual)\n")
        f.write("=" * 72 + "\n")
        for score, review_id, rating_num, rating, reason, text in sorted(results, reverse=True):
            f.write(f"ID: {review_id}\n")
            f.write(f"Score: {score}\n")
            if rating_num is not None:
                f.write(f"Rating_num: {rating_num}\n")
            if isinstance(rating, str) and rating.strip():
                f.write(f"Rating: {rating}\n")
            f.write(f"Motivo: {reason}\n")
            f.write(f"Texto: {text}\n")
            f.write("-" * 72 + "\n")

    print("--- REPORTE SARCASMO POSIBLE ---")
    print(f"Filas analizadas: {total_rows}")
    print(f"Filas omitidas (sin texto): {rows_skipped}")
    print(f"Casos detectados: {sarcasm_detected}")
    print(f"Reporte generado: {report_path}")


if __name__ == "__main__":
    main()
