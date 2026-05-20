import os
import csv
import re
import shutil
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
TEMP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.safe.csv")
BACKUP_PREFIX = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv.safe.bak")

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

REPLACEMENTS = [
    (r"\best ticamente\b", "esteticamente"),
    (r"\best tico\b", "estetico"),
    (r"\best tica\b", "estetica"),
    (r"\best ticos\b", "esteticos"),
    (r"\best ticas\b", "esteticas"),
    (r"\bs no\b", "sino"),
    (r"\bs lo\b", "solo"),
    (r"\bc en\b", "cien"),
    (r"\badem s\b", "ademas"),
    (r"\btambi n\b", "tambien"),
    (r"\bm s\b", "mas"),
    (r"\br pido\b", "rapido"),
    (r"\br pida\b", "rapida"),
    (r"\br pidos\b", "rapidos"),
    (r"\br pidas\b", "rapidas"),
    (r"\bport til\b", "portatil"),
    (r"\bp gina\b", "pagina"),
    (r"\bp ginas\b", "paginas"),
    (r"\bgr fica\b", "grafica"),
    (r"\bgr ficas\b", "graficas"),
    (r"\bgr fico\b", "grafico"),
    (r"\bgr ficos\b", "graficos"),
    (r"\bfranc s\b", "frances"),
    (r"\bal mbrico\b", "alambrico"),
    (r"\bse\s+p de\b", "se puede"),
    (r"\bno s que\b", "no se que"),
    (r"\bsi no s que\b", "si no se que"),
    (r"\bdar s ya\b", "daras ya"),
    (r"\bcu les\b", "cuales"),
    (r"\btendr s\b", "tendras"),
    (r"\bregalar n\b", "regalaran"),
    (r"\bgr ficamente\b", "graficamente"),
    (r"\batr s\b", "atras"),
    (r"\bquiz s\b", "quizas"),
    (r"\bmenls\b", "menos"),
]

TARGET_COLUMN = "final_clean_text"


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


def preserve_case(original, replacement):
    if not original:
        return replacement
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def apply_replacements(text, patterns):
    if not text:
        return text, 0
    new_text = text
    changes = 0
    for pattern, repl in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)

        def _sub(match):
            return preserve_case(match.group(0), repl)

        new_text, count = regex.subn(_sub, new_text)
        changes += count
    return new_text, changes


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"No se encontro el archivo: {SOURCE_FILE}")
        return

    rows, fieldnames, enc_used = read_csv_with_fallback(SOURCE_FILE)
    if not fieldnames or TARGET_COLUMN not in fieldnames:
        print(f"No existe la columna {TARGET_COLUMN}")
        return

    total_changes = 0
    for row in rows:
        original = row.get(TARGET_COLUMN, "")
        updated, changes = apply_replacements(original, REPLACEMENTS)
        row[TARGET_COLUMN] = updated
        total_changes += changes

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_PREFIX}.{timestamp}"
    shutil.copy2(SOURCE_FILE, backup_path)

    with open(TEMP_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(TEMP_FILE, SOURCE_FILE)

    print("Correcciones seguras aplicadas.")
    print(f"Encoding detectado: {enc_used}")
    print(f"Reemplazos totales: {total_changes}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
