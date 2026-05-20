import os
import re
import csv
import shutil
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
BACKUP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv.bak4")
TEMP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.cleaned.csv")

READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

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

WORD_PATTERN = re.compile(r"[A-Za-z\u00c0-\u00ff]+", re.UNICODE)
SPLIT_PATTERN = re.compile(
    r"\b([A-Za-z\u00c0-\u00ff]{1,4})[ \t]+([A-Za-z\u00c0-\u00ff]{2,10})\b",
    re.UNICODE,
)

INSERT_LETTERS = list("aeiournl") + ["\u00f1"]
MIN_WORD_FREQ = 2
MIN_PAIR_FREQ = 2
MIN_CANDIDATE_RATIO = 2.0
MAX_FRAGMENT_FREQ = 3
MIN_CANDIDATE_LEN = 4
VOWELS = set("aeiouáéíóúü")
TEXT_COLUMNS_FOR_VOCAB = [
    "final_clean_text",
    "clean_text",
    "text_after_sarcasm",
]

STOPWORDS = {
    "a",
    "de",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "o",
    "u",
    "que",
    "para",
    "por",
    "con",
    "sin",
    "es",
    "se",
    "al",
    "del",
    "en",
    "mi",
    "tu",
    "su",
    "lo",
    "muy",
    "ya",
    "no",
    "si",
    "me",
    "te",
    "le",
    "yo",
    "este",
    "esta",
    "estos",
    "estas",
    "como",
    "pero",
    "porque",
    "cuando",
    "donde",
    "cual",
}


def read_csv_with_fallback(path):
    last_error = None
    for enc in READ_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, reader.fieldnames, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo leer el CSV: {last_error}")


def is_valid_vocab_token(token):
    if not token:
        return False
    if token in DOMAIN_TERMS:
        return False
    if len(token) < 3:
        return False
    if not any(ch in VOWELS for ch in token):
        return False
    return True


def build_word_counts(rows, columns):
    counts = {}
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            for token in WORD_PATTERN.findall(text):
                if not is_valid_vocab_token(token):
                    continue
                counts[token] = counts.get(token, 0) + 1
    return counts


def build_token_counts(rows, columns):
    counts = {}
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            for token in WORD_PATTERN.findall(text):
                counts[token] = counts.get(token, 0) + 1
    return counts


def build_pair_counts(rows, columns):
    pair_counts = {}
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            tokens = WORD_PATTERN.findall(text)
            for i in range(len(tokens) - 1):
                t1 = tokens[i]
                t2 = tokens[i + 1]
                if t1 in STOPWORDS or t2 in STOPWORDS:
                    continue
                pair_counts[(t1, t2)] = pair_counts.get((t1, t2), 0) + 1
    return pair_counts


def fix_split_words(text, word_counts, pair_counts, token_counts):
    def repl(match):
        t1 = match.group(1)
        t2 = match.group(2)
        t1_lower = t1.lower()
        t2_lower = t2.lower()
        if t1_lower in DOMAIN_TERMS or t2_lower in DOMAIN_TERMS:
            return f"{t1} {t2}"
        if t1_lower in STOPWORDS or t2_lower in STOPWORDS:
            return f"{t1} {t2}"
        if token_counts.get(t1_lower, 0) > MAX_FRAGMENT_FREQ:
            return f"{t1} {t2}"
        if token_counts.get(t2_lower, 0) > MAX_FRAGMENT_FREQ:
            return f"{t1} {t2}"
        pair_count = pair_counts.get((t1_lower, t2_lower), 0)
        if pair_count < MIN_PAIR_FREQ:
            return f"{t1} {t2}"
        candidates = []
        for ch in INSERT_LETTERS:
            candidate = f"{t1_lower}{ch}{t2_lower}"
            freq = word_counts.get(candidate, 0)
            if len(candidate) < MIN_CANDIDATE_LEN:
                continue
            if freq >= MIN_WORD_FREQ and freq >= (pair_count * MIN_CANDIDATE_RATIO):
                candidates.append((candidate, freq))
        if not candidates:
            return f"{t1} {t2}"
        candidates.sort(key=lambda item: item[1], reverse=True)
        best = candidates[0][0]
        if t1[0].isupper():
            return best.capitalize()
        return best

    return SPLIT_PATTERN.sub(repl, text)


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_trailing_period(text):
    stripped = text.rstrip()
    if stripped.endswith(".") and "." not in stripped[:-1]:
        return stripped[:-1].rstrip()
    return text


def clean_text(text, word_counts, pair_counts, token_counts):
    text = str(text).strip()
    if not text:
        return ""

    text = fix_split_words(text, word_counts, pair_counts, token_counts)
    text = strip_trailing_period(text)
    return normalize_spaces(text)


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"No se encontro el archivo: {SOURCE_FILE}")
        return

    rows, fieldnames, enc_used = read_csv_with_fallback(SOURCE_FILE)
    if not fieldnames or "final_clean_text" not in fieldnames:
        print("No existe la columna final_clean_text")
        return

    vocab_columns = [col for col in TEXT_COLUMNS_FOR_VOCAB if col in fieldnames]
    if not vocab_columns:
        vocab_columns = ["final_clean_text"]
    word_counts = build_word_counts(rows, vocab_columns)
    pair_counts = build_pair_counts(rows, vocab_columns)
    token_counts = build_token_counts(rows, vocab_columns)

    for row in tqdm(rows, desc="Corrigiendo", unit="fila"):
        original = row.get("final_clean_text", "")
        row["final_clean_text"] = clean_text(original, word_counts, pair_counts, token_counts)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_FILE
    if os.path.exists(backup_path):
        backup_path = f"{BACKUP_FILE}.{timestamp}"
    shutil.copy2(SOURCE_FILE, backup_path)

    with open(TEMP_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(TEMP_FILE, SOURCE_FILE)

    print("Limpieza conservadora finalizada.")
    print(f"Encoding detectado: {enc_used}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
