import os
import csv
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
REPORT_PATH = os.path.join(BASE_DIR, "reporte_revision_detallada.txt")

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

DOMAIN_TERMS = {
    "hp",
    "dell",
    "lenovo",
    "asus",
    "acer",
    "apple",
    "mac",
    "macbook",
    "macos",
    "intel",
    "ryzen",
    "amd",
    "nvidia",
    "geforce",
    "rtx",
    "gtx",
    "ram",
    "ssd",
    "hdd",
    "nvme",
    "ddr4",
    "ddr5",
    "gb",
    "tb",
    "mhz",
    "ghz",
    "core",
    "i3",
    "i5",
    "i7",
    "i9",
    "celeron",
    "pentium",
    "windows",
    "win",
    "win10",
    "win11",
    "office",
    "laptop",
    "notebook",
    "pc",
    "ips",
    "fhd",
    "uhd",
    "oled",
    "hd",
}

WORD_PATTERN = re.compile(r"[A-Za-z\u00c0-\u00ff]+")
PAIR_PATTERN = re.compile(r"\b([A-Za-z\u00c0-\u00ff]{1,4})\s+([A-Za-z\u00c0-\u00ff]{2,10})\b")
VOWELS = set("aeiouaeiouaeiouaeiouüñáéíóúüAEIOUÁÉÍÓÚÜ")
INSERT_LETTERS = list("aeiouáéíóúü") + ["ñ"]

VOCAB_COLUMNS = [
    "final_clean_text",
    "clean_text",
    "text_after_sarcasm",
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


def build_vocab(rows, columns):
    vocab = set()
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            for token in WORD_PATTERN.findall(text):
                if token in DOMAIN_TERMS:
                    continue
                if len(token) < 3:
                    continue
                vocab.add(token)
    return vocab


def find_missing_letter_pairs(text, vocab):
    findings = []
    for m in PAIR_PATTERN.finditer(text):
        t1 = m.group(1)
        t2 = m.group(2)
        t1_lower = t1.lower()
        t2_lower = t2.lower()
        if t1_lower in DOMAIN_TERMS or t2_lower in DOMAIN_TERMS:
            continue
        for ch in INSERT_LETTERS:
            candidate = f"{t1_lower}{ch}{t2_lower}"
            if candidate in vocab:
                findings.append((t1, t2, candidate))
                break
    return findings


def find_no_vowel_tokens(text):
    tokens = []
    for token in WORD_PATTERN.findall(text):
        if len(token) >= 4 and not any(ch in VOWELS for ch in token):
            if token.lower() not in DOMAIN_TERMS:
                tokens.append(token)
    return tokens


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontro el archivo: {CSV_PATH}")
        return

    rows, fieldnames, used_enc = read_csv_with_fallback(CSV_PATH)
    if not fieldnames or "final_clean_text" not in fieldnames:
        print("Falta columna final_clean_text")
        return

    vocab_columns = [c for c in VOCAB_COLUMNS if c in fieldnames]
    if not vocab_columns:
        vocab_columns = ["final_clean_text"]
    vocab = build_vocab(rows, vocab_columns)

    issue_rows = 0
    pair_counter = Counter()
    no_vowel_counter = Counter()

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Encoding usado: {used_enc}\n")
        f.write(f"Total filas: {len(rows)}\n\n")
        f.write("Filas con posibles letras faltantes en final_clean_text:\n")

        for idx, row in enumerate(rows, start=2):
            text = str(row.get("final_clean_text", ""))
            id_review = row.get("id_review", "")

            pair_findings = find_missing_letter_pairs(text, vocab)
            no_vowel = find_no_vowel_tokens(text)

            if pair_findings or no_vowel:
                issue_rows += 1
                f.write("-" * 80 + "\n")
                f.write(f"Linea CSV: {idx} | id_review: {id_review}\n")
                if pair_findings:
                    f.write("Posibles cortes detectados:\n")
                    for t1, t2, cand in pair_findings[:5]:
                        f.write(f"  {t1} {t2} -> {cand}\n")
                        pair_counter[(t1.lower(), t2.lower())] += 1
                if no_vowel:
                    f.write("Tokens sin vocales:\n")
                    for token in no_vowel[:5]:
                        f.write(f"  {token}\n")
                        no_vowel_counter[token.lower()] += 1
                f.write(f"Texto: {text}\n")

        f.write("\nResumen:\n")
        f.write(f"Filas con posibles errores: {issue_rows}\n")
        f.write("Top 20 cortes mas frecuentes:\n")
        for (t1, t2), count in pair_counter.most_common(20):
            f.write(f"  {t1} {t2}: {count}\n")
        f.write("Top 20 tokens sin vocales:\n")
        for token, count in no_vowel_counter.most_common(20):
            f.write(f"  {token}: {count}\n")

    print("Revision detallada generada:")
    print(REPORT_PATH)
    print(f"Filas con posibles errores: {issue_rows}")


if __name__ == "__main__":
    main()
