import os
import csv
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

WORD_PATTERN = re.compile(r"[A-Za-z\u00c0-\u00ff]+")
PAIR_PATTERN = re.compile(r"\b([A-Za-z\u00c0-\u00ff]{1,4})\s+([A-Za-z\u00c0-\u00ff]{2,10})\b")

VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
SUSPICIOUS_CHARS = ["�", "Ã", "Â", "£", "¢", "¤", "¥", "¦", "¨", "ª", "º", "¡", "¿"]


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


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontro el archivo: {CSV_PATH}")
        return

    rows, used_enc = read_csv_with_fallback(CSV_PATH)
    if not rows:
        print("CSV vacio.")
        return
    if "final_clean_text" not in rows[0]:
        print("Falta columna final_clean_text")
        return

    pair_counts = Counter()
    no_vowel_counts = Counter()
    suspicious_counts = Counter()
    example_pairs = []
    example_suspicious = []

    for row in rows:
        text = str(row.get("final_clean_text", ""))
        for ch in SUSPICIOUS_CHARS:
            if ch in text:
                suspicious_counts[ch] += text.count(ch)
                if len(example_suspicious) < 5:
                    example_suspicious.append(text)
        for m in PAIR_PATTERN.finditer(text):
            pair = (m.group(1).lower(), m.group(2).lower())
            pair_counts[pair] += 1
            if len(example_pairs) < 5:
                example_pairs.append(m.group(0))
        for token in WORD_PATTERN.findall(text):
            if len(token) >= 4 and not any(v in token for v in VOWELS):
                no_vowel_counts[token.lower()] += 1

    print("Encoding usado:", used_enc)
    print("Total filas:", len(rows))

    print("\nCaracteres sospechosos encontrados:")
    for ch in SUSPICIOUS_CHARS:
        if suspicious_counts.get(ch, 0):
            print(f"  {repr(ch)}: {suspicious_counts[ch]}")

    print("\nTop 20 pares sospechosos (fragmentos separados):")
    for (t1, t2), count in pair_counts.most_common(20):
        print(f"{t1} {t2}: {count}")

    print("\nTop 20 tokens sin vocales (>=4 letras):")
    for token, count in no_vowel_counts.most_common(20):
        print(f"{token}: {count}")

    if example_pairs:
        print("\nEjemplos de pares detectados:")
        for ex in example_pairs:
            print(f"  {ex}")

    if example_suspicious:
        print("\nEjemplos con caracteres sospechosos:")
        for ex in example_suspicious:
            print(f"  {ex[:140]}")


if __name__ == "__main__":
    main()
