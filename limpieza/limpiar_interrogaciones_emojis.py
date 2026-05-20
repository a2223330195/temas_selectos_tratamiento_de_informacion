import re
import time
import unicodedata
from pathlib import Path
import shutil

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

PALABRAS_INTERROGATIVAS = [
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    "cuanto", "cuánto", "por qué", "trae", "incluye", "sirve", "funciona",
    "garantia", "garantía", "puedo", "tiene",
]

PALABRAS_AFIRMACION = [
    "excelente", "recomendable", "bueno", "buen", "bonito", "hermoso",
    "malo", "pesimo", "pésimo", "perfecto", "encanto", "encantó", "gusta",
]

WORD_PATTERN = re.compile(r"[a-z]+", flags=re.IGNORECASE)


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path} con las codificaciones probadas.")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return stripped.lower()


def build_word_set(words: list[str]) -> set[str]:
    return {normalize_text(word) for word in words}


def has_interrogative(text_norm: str, tokens: set[str], interrogatives: set[str]) -> bool:
    if tokens.intersection(interrogatives):
        return True
    return bool(re.search(r"\bpor\s+que\b", text_norm))


def clean_question_marks(text: str) -> str:
    if not isinstance(text, str):
        return ""
    raw = text
    if "?" not in raw and "¿" not in raw:
        cleaned = raw.strip()
        return "" if re.fullmatch(r"[.\s]+", cleaned or "") else cleaned

    text_norm = normalize_text(raw)
    tokens = set(WORD_PATTERN.findall(text_norm))
    word_count = len(tokens)

    interrogatives = build_word_set(PALABRAS_INTERROGATIVAS)
    affirmations = build_word_set(PALABRAS_AFIRMACION)

    if has_interrogative(text_norm, tokens, interrogatives):
        cleaned = raw.strip()
        return "" if re.fullmatch(r"[.\s]+", cleaned or "") else cleaned

    if word_count <= 3 or tokens.intersection(affirmations):
        replaced = re.sub(r"[?¿]", ".", raw)
        cleaned = replaced.strip()
        return "" if re.fullmatch(r"[.\s]+", cleaned or "") else cleaned

    replaced = re.sub(r"[?¿]{3,}", ".", raw)
    cleaned = replaced.strip()
    return "" if re.fullmatch(r"[.\s]+", cleaned or "") else cleaned


def main() -> None:
    df = load_csv(CSV_PATH)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna final_clean_text.")

    tqdm.pandas(desc="Limpieza")
    df["final_clean_text"] = df["final_clean_text"].progress_apply(clean_question_marks)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".bak.{timestamp}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("--- LIMPIEZA INTERROGACIONES ---")
    print(f"Filas procesadas: {len(df)}")
    print(f"Backup: {backup_path}")
    print(f"CSV actualizado: {CSV_PATH}")


if __name__ == "__main__":
    main()
