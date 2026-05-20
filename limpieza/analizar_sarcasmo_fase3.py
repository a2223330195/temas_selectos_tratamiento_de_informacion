import os
import csv
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]


def read_csv_with_fallback(path):
    last_error = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo leer el CSV: {last_error}")


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "si", "yes"}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontro el archivo: {CSV_PATH}")
        return

    rows, enc = read_csv_with_fallback(CSV_PATH)
    if not rows:
        print("CSV vacio.")
        return

    total = len(rows)
    possible_true = 0
    action_final_counts = Counter()
    review_status_counts = Counter()
    action_counts = Counter()

    same_final_updated = 0
    same_final_after = 0
    same_final_pre = 0
    updated_field_present = 0

    for row in rows:
        possible = row.get("possible_sarcasm", "")
        if is_true(possible):
            possible_true += 1

        action_final = row.get("sarcasm_action_final", "")
        if action_final:
            action_final_counts[action_final] += 1

        review_status = row.get("sarcasm_review_status", "")
        if review_status:
            review_status_counts[review_status] += 1

        action = row.get("sarcasm_action", "")
        if action:
            action_counts[action] += 1

        final_text = row.get("final_clean_text", "")
        updated_text = row.get("clean_text_updated_after_sarcasm", "")
        after_text = row.get("text_after_sarcasm", "")
        pre_text = row.get("clean_text_original_pre_sarcasm", "")

        if updated_text:
            updated_field_present += 1
        if final_text and updated_text and final_text == updated_text:
            same_final_updated += 1
        if final_text and after_text and final_text == after_text:
            same_final_after += 1
        if final_text and pre_text and final_text == pre_text:
            same_final_pre += 1

    print("Encoding usado:", enc)
    print("Total filas:", total)
    print("possible_sarcasm=True:", possible_true)
    print("\nConteo sarcasm_action:")
    for key, count in action_counts.most_common():
        print(f"  {key}: {count}")
    print("\nConteo sarcasm_review_status:")
    for key, count in review_status_counts.most_common():
        print(f"  {key}: {count}")
    print("\nConteo sarcasm_action_final:")
    for key, count in action_final_counts.most_common():
        print(f"  {key}: {count}")

    print("\nComparacion de final_clean_text:")
    print("  final == clean_text_updated_after_sarcasm:", same_final_updated)
    print("  final == text_after_sarcasm:", same_final_after)
    print("  final == clean_text_original_pre_sarcasm:", same_final_pre)
    print("  filas con clean_text_updated_after_sarcasm no vacio:", updated_field_present)


if __name__ == "__main__":
    main()
