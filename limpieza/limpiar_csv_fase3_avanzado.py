import os
import re
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
TEMP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.cleaned.csv")
BACKUP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv.bak")

READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

MOJIBAKE_MAP = {
    "\u00c3\u00a1": "\u00e1",
    "\u00c3\u00a9": "\u00e9",
    "\u00c3\u00ad": "\u00ed",
    "\u00c3\u00b3": "\u00f3",
    "\u00c3\u00ba": "\u00fa",
    "\u00c3\u00b1": "\u00f1",
    "\u00c3\u00bc": "\u00fc",
    "\u00c3\u0081": "\u00c1",
    "\u00c3\u0089": "\u00c9",
    "\u00c3\u008d": "\u00cd",
    "\u00c3\u0093": "\u00d3",
    "\u00c3\u009a": "\u00da",
    "\u00c3\u0091": "\u00d1",
    "\u00c3\u009c": "\u00dc",
    "\u00e2\u0080\u0099": "\u2019",
    "\u00e2\u0080\u009c": "\"",
    "\u00e2\u0080\u009d": "\"",
    "\u00e2\u0080\u0094": "-",
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u00a6": "...",
    "\u00e2\u0080\u00a2": "-",
    "\u00c2": "",
}

SUSPICIOUS_MARKERS = [
    "\ufffd",
    "\u00c3",
    "\u00c2",
    "\u00e2",
    "\u00a3",
    "\u00a2",
    "\u00a4",
    "\u00a5",
    "\u00a6",
    "\u00a8",
    "\u00aa",
    "\u00ba",
    "\u00a1",
    "\u00bf",
]

SPANISH_REWARD_CHARS = "\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc\u00c1\u00c9\u00cd\u00d3\u00da\u00d1\u00dc"
SPANISH_WORDS = [
    "que",
    "para",
    "con",
    "muy",
    "bueno",
    "buena",
    "excelente",
    "producto",
    "precio",
    "calidad",
    "equipo",
    "recomiendo",
    "recomendado",
    "laptop",
    "computadora",
    "rapido",
    "ligera",
    "pantalla",
    "funciona",
    "funcional",
    "trabajo",
    "escuela",
]


def read_csv_with_fallback(path):
    last_error = None
    for encoding in READ_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"No se pudo leer el CSV: {last_error}")


def score_text(value):
    if not value:
        return 0
    text = value
    penalty = 0
    for marker in SUSPICIOUS_MARKERS:
        penalty += text.count(marker)
    reward = 0
    for ch in SPANISH_REWARD_CHARS:
        reward += text.count(ch)
    lower = text.lower()
    for word in SPANISH_WORDS:
        if re.search(rf"\\b{re.escape(word)}\\b", lower):
            reward += 2
    return reward - penalty


def try_recode(value, enc_in, enc_out):
    try:
        return value.encode(enc_in).decode(enc_out)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def normalize_whitespace(text):
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def apply_mojibake_map(text):
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    return text


def choose_best(original, candidates):
    best_text = original
    best_score = score_text(original)
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_score = score_text(candidate)
        if candidate_score > best_score:
            best_text = candidate
            best_score = candidate_score
    return best_text


def sanitize_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\ufeff", "").replace("\ufffd", "")

    candidates = []
    for enc_in in ["latin1", "cp1252", "cp850", "cp437"]:
        candidates.append(try_recode(text, enc_in, "utf-8"))
        for enc_out in ["latin1", "cp1252", "cp850", "cp437"]:
            if enc_out == enc_in:
                continue
            candidates.append(try_recode(text, enc_in, enc_out))

    text = choose_best(text, candidates)
    text = apply_mojibake_map(text)
    text = normalize_whitespace(text)
    return text


def clean_dataframe(df):
    changed = 0
    for col in df.columns:
        if df[col].dtype == object:
            cleaned_col = []
            for value in df[col].tolist():
                cleaned_value = sanitize_text(value)
                if cleaned_value != value:
                    changed += 1
                cleaned_col.append(cleaned_value)
            df[col] = cleaned_col
    return changed


def main():
    source_path = BACKUP_FILE if os.path.exists(BACKUP_FILE) else SOURCE_FILE
    if not os.path.exists(source_path):
        print(f"No se encontro el archivo: {source_path}")
        return

    df = read_csv_with_fallback(source_path)
    changed = clean_dataframe(df)

    df.to_csv(TEMP_FILE, index=False, encoding="utf-8-sig")

    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(SOURCE_FILE, BACKUP_FILE)
    os.replace(TEMP_FILE, SOURCE_FILE)

    print("CSV limpiado. Backup en:")
    print(BACKUP_FILE)
    print(f"Celdas modificadas: {changed}")


if __name__ == "__main__":
    main()
