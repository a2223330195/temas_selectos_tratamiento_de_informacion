import os
import csv
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

SAR_COLUMNS = [
    "possible_sarcasm",
    "sarcasm_reason",
    "sarcasm_action",
    "sarcasm_review_status",
    "text_after_sarcasm",
    "clean_text_original_pre_sarcasm",
    "sarcasm_manual_decision",
    "sarcasm_processing_notes",
    "sarcasm_action_final",
    "clean_text_updated_after_sarcasm",
]


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


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "si", "yes"}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontro el archivo: {CSV_PATH}")
        return

    rows, fieldnames, enc = read_csv_with_fallback(CSV_PATH)
    if not rows:
        print("CSV vacio.")
        return

    missing_cols = [c for c in SAR_COLUMNS if c not in fieldnames]
    if missing_cols:
        print("Faltan columnas:")
        for c in missing_cols:
            print(f"  {c}")
        return

    total = len(rows)
    non_empty_counts = Counter()
    action_final_counts = Counter()
    action_counts = Counter()
    review_status_counts = Counter()

    mismatch_possible_review = 0
    mismatch_possible_action = 0

    for row in rows:
        possible = row.get("possible_sarcasm", "")
        review_status = row.get("sarcasm_review_status", "")
        action = row.get("sarcasm_action", "")
        action_final = row.get("sarcasm_action_final", "")

        for col in SAR_COLUMNS:
            if str(row.get(col, "")).strip():
                non_empty_counts[col] += 1

        if action:
            action_counts[action] += 1
        if review_status:
            review_status_counts[review_status] += 1
        if action_final:
            action_final_counts[action_final] += 1

        if is_true(possible):
            if review_status != "requiere revisión manual":
                mismatch_possible_review += 1
            if action != "revisar":
                mismatch_possible_action += 1
        else:
            if review_status and review_status != "sin revisión":
                mismatch_possible_review += 1
            if action and action != "conservar":
                mismatch_possible_action += 1

    print("Encoding usado:", enc)
    print("Total filas:", total)

    print("\nColumnas sarcasmo con valores no vacios:")
    for col in SAR_COLUMNS:
        print(f"  {col}: {non_empty_counts[col]}")

    print("\nConteo sarcasm_action:")
    for k, v in action_counts.most_common():
        print(f"  {k}: {v}")

    print("\nConteo sarcasm_review_status:")
    for k, v in review_status_counts.most_common():
        print(f"  {k}: {v}")

    print("\nConteo sarcasm_action_final:")
    for k, v in action_final_counts.most_common():
        print(f"  {k}: {v}")

    print("\nInconsistencias vs possible_sarcasm:")
    print(f"  review_status inconsistentes: {mismatch_possible_review}")
    print(f"  sarcasm_action inconsistentes: {mismatch_possible_action}")


if __name__ == "__main__":
    main()
