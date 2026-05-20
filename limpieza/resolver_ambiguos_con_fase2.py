import os
import csv
import re
import shutil
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase2.csv")
PHASE3_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
TEMP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.ambfix.csv")
BACKUP_PREFIX = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv.ambfix.bak")

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

WORD_PATTERN = re.compile(r"[A-Za-z\u00c0-\u00ff]+")
VOWELS = set("aeiouAEIOUaeiouaeiouaeiouaeiou\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00d1\u00f1")

AMBIGUOUS_PATTERNS = [
    re.compile(r"\bs\s+que\b", flags=re.IGNORECASE),
    re.compile(r"\bn\s+se\b", flags=re.IGNORECASE),
    re.compile(r"\bs\s+la\b", flags=re.IGNORECASE),
    re.compile(r"\bs\s+ya\b", flags=re.IGNORECASE),
    re.compile(r"\bs\s+pero\b", flags=re.IGNORECASE),
    re.compile(r"\by\s+un\b", flags=re.IGNORECASE),
]

ALLOWED_NO_VOWEL = {"mgsv", "spss", "ctrl", "ltsc", "dlls", "fpss", "sdcs", "nmms"}


def read_csv_with_fallback(path):
    last_error = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, reader.fieldnames, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo leer el CSV: {last_error}")


def has_ambiguous_pattern(text):
    for pattern in AMBIGUOUS_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_suspicious_tokens(text):
    for token in WORD_PATTERN.findall(text):
        lower = token.lower()
        if lower in ALLOWED_NO_VOWEL:
            continue
        if len(token) >= 4 and not any(ch in VOWELS for ch in token):
            return True
    return False


def main():
    if not os.path.exists(PHASE2_PATH):
        print(f"No se encontro el archivo: {PHASE2_PATH}")
        return
    if not os.path.exists(PHASE3_PATH):
        print(f"No se encontro el archivo: {PHASE3_PATH}")
        return

    phase2_rows, phase2_fields, enc2 = read_csv_with_fallback(PHASE2_PATH)
    phase3_rows, phase3_fields, enc3 = read_csv_with_fallback(PHASE3_PATH)

    if "id_review" not in phase2_fields or "final_clean_text" not in phase2_fields:
        print("Faltan columnas en fase2 (id_review/final_clean_text)")
        return
    if "id_review" not in phase3_fields or "final_clean_text" not in phase3_fields:
        print("Faltan columnas en fase3 (id_review/final_clean_text)")
        return

    phase2_map = {}
    for row in phase2_rows:
        key = row.get("id_review", "")
        if key:
            phase2_map[key] = row.get("final_clean_text", "")

    total_updates = 0
    total_checked = 0

    for row in phase3_rows:
        text = str(row.get("final_clean_text", ""))
        if not text:
            continue
        total_checked += 1
        if not (has_ambiguous_pattern(text) or has_suspicious_tokens(text)):
            continue
        key = row.get("id_review", "")
        if not key:
            continue
        replacement = phase2_map.get(key)
        if replacement and replacement != text:
            row["final_clean_text"] = replacement
            total_updates += 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_PREFIX}.{timestamp}"
    shutil.copy2(PHASE3_PATH, backup_path)

    with open(TEMP_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase3_fields)
        writer.writeheader()
        writer.writerows(phase3_rows)

    os.replace(TEMP_FILE, PHASE3_PATH)

    print("Ambiguos resueltos usando fase2.")
    print(f"Encoding fase2: {enc2}")
    print(f"Encoding fase3: {enc3}")
    print(f"Filas revisadas: {total_checked}")
    print(f"Filas actualizadas: {total_updates}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
