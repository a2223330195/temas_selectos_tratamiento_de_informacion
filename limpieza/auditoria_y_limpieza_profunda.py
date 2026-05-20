import re
import shutil
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from spellchecker import SpellChecker
except ImportError as exc:
    raise SystemExit(
        "Falta pyspellchecker. Instala con: pip install pyspellchecker"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

ENCODING_MAP = {
    "ms": "más",
    "diseo": "diseño",
    "batera": "batería",
    "ao": "año",
    "anio": "año",
    "anos": "años",
    "manana": "mañana",
    "nino": "niño",
    "senal": "señal",
}

SHORT_EXPANSIONS = {
    "k": "que",
    "ok": "esta bien",
}

PROTECTED_TERMS = {
    "hp", "dell", "lenovo", "acer", "asus", "msi", "ram", "ssd", "hdd",
    "gpu", "cpu", "intel", "amd", "nvidia", "rtx", "gtx", "usb", "hdmi",
    "windows", "mac", "macbook", "office", "linux", "wifi", "bluetooth",
    "ryzen", "i3", "i5", "i7", "i9",
}



def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path} con las codificaciones probadas.")


def preserve_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def apply_encoding_map(text: str) -> str:
    for wrong, right in ENCODING_MAP.items():
        pattern = rf"\b{re.escape(wrong)}\b"
        text = re.sub(pattern, lambda m: preserve_case(right, m.group(0)), text, flags=re.IGNORECASE)
    return text


def expand_short_tokens(text: str) -> str:
    for short, long_form in SHORT_EXPANSIONS.items():
        pattern = rf"\b{re.escape(short)}\b"
        text = re.sub(pattern, long_form, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if text.endswith(".") and not text.endswith("..."):
        text = text[:-1].rstrip()
    return text


def spellcheck_text(text: str, spell: SpellChecker) -> tuple[str, int]:
    token_pattern = r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b"
    tokens = re.findall(token_pattern, text)
    unique_tokens = {t for t in tokens if t}
    corrections = {}

    for token in unique_tokens:
        lower = token.lower()
        if lower in PROTECTED_TERMS:
            continue
        if token.isupper() or any(ch.isdigit() for ch in token):
            continue
        if len(lower) <= 2:
            continue
        if lower in spell:
            continue
        suggestion = spell.correction(lower)
        if suggestion and suggestion != lower:
            corrections[lower] = suggestion

    if not corrections:
        return text, 0

    def replace_match(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in corrections:
            return preserve_case(corrections[lower], word)
        return word

    new_text = re.sub(token_pattern, replace_match, text)
    return new_text, len(corrections)




def main() -> None:
    df = load_csv(CSV_PATH)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna final_clean_text.")

    if "final_clean_text_original" not in df.columns:
        df["final_clean_text_original"] = df["final_clean_text"]

    spell = SpellChecker(language="es")
    spell.word_frequency.load_words(PROTECTED_TERMS)

    text_changes = 0
    spell_changes = 0
    rows_processed = 0
    rows_skipped = 0

    total_rows = len(df)
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Procesando", unit="fila"):
        base_text = None
        for col in ("text_after_sarcasm", "final_clean_text", "clean_text", "raw_text"):
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                base_text = val
                break
        if not base_text:
            rows_skipped += 1
            continue

        rows_processed += 1

        text = apply_encoding_map(base_text)
        text = expand_short_tokens(text)
        text, corrections_count = spellcheck_text(text, spell)
        text = normalize_text(text)

        if text != row.get("final_clean_text"):
            df.at[idx, "final_clean_text"] = text
            text_changes += 1
        if corrections_count:
            spell_changes += 1


    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".bak.{timestamp}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("--- REPORTE DE AUDITORIA ---")
    print(f"Filas procesadas: {rows_processed}")
    print(f"Filas omitidas (sin texto): {rows_skipped}")
    print(f"Filas con texto corregido: {text_changes}")
    print(f"Filas con correccion ortografica: {spell_changes}")
    print(f"Backup creado: {backup_path}")


if __name__ == "__main__":
    main()
