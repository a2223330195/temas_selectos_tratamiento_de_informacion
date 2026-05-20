import os
import csv
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv")
TEMP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.fixed.csv")
BACKUP_FILE = os.path.join(BASE_DIR, "corpus_limpio_final_fase3.csv.bak2")
REPORT_FILE = os.path.join(BASE_DIR, "reporte_reparaciones_auto.txt")

READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp850", "cp437"]

AUTO_DETECT = True
MIN_PAIR_FREQ = 3
MIN_WORD_FREQ = 3

TEXT_COLUMNS = [
    "raw_text",
    "clean_text",
    "final_clean_text",
    "spell_corrected_text",
    "spell_corrected_text_v2",
    "spelling_notes",
    "audit_notes",
    "sarcasm_reason",
    "sarcasm_processing_notes",
    "fase2_notes",
    "text_after_sarcasm",
    "clean_text_original_pre_sarcasm",
]

STOPWORDS = {
    "a", "e", "y", "o", "u", "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "que", "para", "por", "con", "sin", "es", "se", "al", "del", "en", "mi", "tu", "su",
    "lo", "muy", "ya", "no", "si", "me", "te", "le", "yo", "ella", "ellos", "ellas",
    "este", "esta", "estos", "estas", "como", "pero", "porque", "cuando", "donde", "cual",
}

TOKEN_PATTERN = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+")
LETTERS_TO_TRY = list("aeiounrlmst") + ["ñ"]

REPLACEMENTS = [
    (r"\br pida\b", "rapida"),
    (r"\br pido\b", "rapido"),
    (r"\bm s\b", "mas"),
    (r"\bpr ctica\b", "practica"),
    (r"\bpr ctico\b", "practico"),
    (r"\bf cil\b", "facil"),
    (r"\best bien\b", "esta bien"),
    (r"\bb sico\b", "basico"),
    (r"\bb sicos\b", "basicos"),
    (r"\bb sica\b", "basica"),
    (r"\bb sicas\b", "basicas"),
    (r"\bdem s\b", "demas"),
    (r"\bgr ficos\b", "graficos"),
    (r"\bgr fica\b", "grafica"),
    (r"\bus ndola\b", "usandola"),
    (r"\best tica\b", "estetica"),
    (r"\bm quina\b", "maquina"),
    (r"\bm ximo\b", "maximo"),
    (r"\bc mara\b", "camara"),
    (r"\btama o\b", "tamano"),
    (r"\bs per\b", "super"),
    (r"\bingl s\b", "ingles"),
    (r"\bofim tica\b", "ofimatica"),
    (r"\bpl stico\b", "plastico"),
    (r"\bim genes\b", "imagenes"),
    (r"\best ticos\b", "esteticos"),
    (r"\best s\b", "estos"),
    (r"\bfr gil\b", "fragil"),
    (r"\bf brica\b", "fabrica"),
    (r"\bus ndolo\b", "usandolo"),
    (r"\b([a-z]{2,})ci n\b", r"\1cion"),
    (r"\b([a-z]{2,})si n\b", r"\1sion"),
]

TARGET_COLUMNS = ["final_clean_text", "clean_text", "text_after_sarcasm"]


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
            continue
    raise RuntimeError(f"No se pudo leer el CSV: {last_error}")


def apply_replacements(text, replacement_regex):
    if not text:
        return text, 0
    new_text = text
    changes = 0
    for regex, repl in replacement_regex:
        new_text, count = regex.subn(repl, new_text)
        changes += count
    if new_text != text:
        new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text, changes


def build_word_counts(rows, columns):
    word_counts = {}
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            for token in TOKEN_PATTERN.findall(text):
                word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts


def build_pair_counts(rows, columns):
    pair_counts = {}
    for row in rows:
        for col in columns:
            text = str(row.get(col, "")).lower()
            tokens = TOKEN_PATTERN.findall(text)
            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]
                if len(t1) > 15 or len(t2) > 8:
                    continue
                if t1 in STOPWORDS or t2 in STOPWORDS:
                    continue
                pair = (t1, t2)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def build_auto_replacements(rows, columns):
    word_counts = build_word_counts(rows, columns)
    pair_counts = build_pair_counts(rows, columns)

    auto_replacements = []
    for (t1, t2), count in pair_counts.items():
        if count < MIN_PAIR_FREQ:
            continue
        best_candidate = None
        best_freq = 0
        for ch in LETTERS_TO_TRY:
            candidate = f"{t1}{ch}{t2}"
            freq = word_counts.get(candidate, 0)
            if freq >= MIN_WORD_FREQ and freq > best_freq:
                best_candidate = candidate
                best_freq = freq
        if best_candidate:
            pattern = rf"\b{re.escape(t1)}\s+{re.escape(t2)}\b"
            auto_replacements.append((pattern, best_candidate, count, best_freq))
    return auto_replacements


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"No se encontro el archivo: {SOURCE_FILE}")
        return

    rows, fieldnames, enc_used = read_csv_with_fallback(SOURCE_FILE)
    if not fieldnames:
        print("CSV sin encabezados.")
        return

    columns_to_fix = [col for col in TARGET_COLUMNS if col in fieldnames]
    if not columns_to_fix:
        print("No se encontraron columnas objetivo para reparar.")
        return

    auto_replacements = []
    if AUTO_DETECT:
        auto_replacements = build_auto_replacements(rows, columns_to_fix)

    replacement_regex = [(re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in REPLACEMENTS]
    replacement_regex.extend(
        (re.compile(pat, flags=re.IGNORECASE), repl)
        for pat, repl, _, _ in auto_replacements
    )

    total_changes = 0
    for row in rows:
        for col in columns_to_fix:
            original = row.get(col, "")
            fixed, changes = apply_replacements(original, replacement_regex)
            row[col] = fixed
            total_changes += changes

    with open(TEMP_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(SOURCE_FILE, BACKUP_FILE)
    os.replace(TEMP_FILE, SOURCE_FILE)

    print("Reparacion aplicada.")
    print(f"Encoding detectado: {enc_used}")
    print(f"Total de reemplazos: {total_changes}")
    print(f"Backup: {BACKUP_FILE}")

    if auto_replacements:
        auto_replacements.sort(key=lambda x: (-x[2], x[0]))
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("Auto-reemplazos detectados (patron -> reemplazo | pares | palabra):\n")
            for pat, repl, pair_count, word_freq in auto_replacements:
                f.write(f"{pat} -> {repl} | pares: {pair_count} | palabra: {word_freq}\n")
        print(f"Reporte: {REPORT_FILE}")


if __name__ == "__main__":
    main()
