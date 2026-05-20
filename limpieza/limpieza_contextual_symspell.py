import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from symspellpy import SymSpell, Verbosity
except ImportError as exc:
    raise SystemExit(
        "Falta symspellpy. Instala con: pip install symspellpy"
    ) from exc

try:
    from wordfreq import top_n_list, zipf_frequency
except ImportError as exc:
    raise SystemExit(
        "Falta wordfreq. Instala con: pip install wordfreq"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

TECH_TERMS = {
    "laptop", "notebook", "ultrabook", "macbook", "windows", "office",
    "hp", "dell", "lenovo", "acer", "asus", "msi", "ram", "ssd", "hdd",
    "gpu", "cpu", "intel", "amd", "nvidia", "rtx", "gtx", "usb", "hdmi",
    "wifi", "bluetooth", "ryzen", "i3", "i5", "i7", "i9",
    "autocad", "roblox", "minecraft", "fortnite", "valorant", "capcut",
    "matlab", "proteus", "canva", "tsuru", "witcher", "shaders",
    "steam", "office365", "powerpoint", "excel", "word",
}

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
    "sta": "esta",
}

CONFUSABLES = [
    (r"\bapto\b", "laptop"),
    (r"\blap\b", "laptop"),
    (r"\bremiendo\b", "recomiendo"),
    (r"\bsuero\b", "supero"),
    (r"\btizne\b", "tiene"),
    (r"\btiine\b", "tiene"),
    (r"\bsirle\b", "sirve"),
    (r"\buñi\b", "uni"),
    (r"\blord\b", "word"),
    (r"\bme\s+precio\b", "me parecio"),
    (r"\btambi\s+n\b", "tambien"),
    (r"\btambo\s+n\b", "tambien"),
    (r"\bbiene\b", "viene"),
    (r"\be\s+comprado\b", "he comprado"),
    (r"\bexelente\b", "excelente"),
    (r"\bexecelente\b", "excelente"),
    (r"\bhome\s+oficie\b", "home office"),
    (r"\bregqlo\b", "regalo"),
    (r"\bbatroas\b", "batallas"),
]

CHAT_EXPANSIONS = {
    "q": "que",
    "k": "que",
    "x": "por",
    "pq": "porque",
    "xq": "porque",
    "tmb": "tambien",
    "tb": "tambien",
}

VOWELS = ("a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü")
ACCENTED_VOWELS = {"á", "é", "í", "ó", "ú"}
STOP_MERGE_SINGLE = {"y", "e", "o", "u", "a", "d", "x", "q", "k", "i"}
STOP_MERGE_WORDS = {
    "de", "del", "la", "el", "los", "las", "y", "o", "a", "en", "por",
    "para", "que", "con", "sin", "al", "su", "sus", "mi", "mis", "lo",
    "me", "te", "se", "le", "les", "nos", "ya",
}
ESTA_VERB_NEXT = {
    "bien", "mal", "perfecto", "perfecta", "perfectos", "perfectas",
    "genial", "en", "muy", "ok", "super", "súper", "listo", "lista",
    "bonito", "bonita", "correcto", "correcta", "completo", "completa",
    "lindo", "linda", "hermoso", "hermosa", "excelente", "increíble",
    "disponible",
}
MIN_ZIPF_PART = 3.0
MIN_ZIPF_JOIN_STRICT = 2.2
MIN_ZIPF_JOIN_LOOSE = 1.6
MIN_ZIPF_JOIN_GAP = 0.5
MIN_ZIPF_CORPUS = 2.5
MIN_ZIPF_KEEP = 2.2
MIN_ZIPF_GAIN = 0.9

WORD_PATTERN = re.compile(r"[a-záéíóúüñ]+", flags=re.IGNORECASE)


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path} con las codificaciones probadas.")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def preserve_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def apply_encoding_map(text: str) -> str:
    for wrong, right in ENCODING_MAP.items():
        pattern = rf"\b{re.escape(wrong)}\b"
        text = re.sub(pattern, lambda m: preserve_case(right, m.group(0)), text, flags=re.IGNORECASE)
    return text


def has_accent(word: str) -> bool:
    return any(ch in ACCENTED_VOWELS for ch in word)


def is_common_word(word: str) -> bool:
    return zipf_frequency(word, "es") >= MIN_ZIPF_PART


def consonant_skeleton(word: str) -> str:
    return re.sub(r"[aeiouáéíóúü]", "", word.lower())


def best_vowel_join(left: str, right: str) -> Tuple[Optional[str], float, float]:
    best_word = None
    best_score = 0.0
    second_score = 0.0
    best_accent_word = None
    best_accent_score = 0.0

    for vowel in VOWELS:
        candidate = f"{left}{vowel}{right}"
        score = zipf_frequency(candidate, "es")
        if score > best_score:
            second_score = best_score
            best_score = score
            best_word = candidate
        elif score > second_score:
            second_score = score

        if has_accent(candidate) and score > best_accent_score:
            best_accent_score = score
            best_accent_word = candidate

    if best_word and best_accent_word and not has_accent(best_word):
        if best_accent_score >= best_score - 0.25:
            best_word = best_accent_word
            best_score = best_accent_score

    return best_word, best_score, second_score


def should_merge_parts(left: str, right: str) -> bool:
    if left in STOP_MERGE_WORDS or right in STOP_MERGE_WORDS:
        return False
    if len(left) == 1 and left not in STOP_MERGE_SINGLE:
        return True
    if len(right) == 1 and right not in STOP_MERGE_SINGLE:
        return True
    if is_common_word(left) and is_common_word(right):
        return False
    return True

def merge_broken_words(text: str) -> str:
    parts = re.findall(r"[a-záéíóúüñ]+|[^a-záéíóúüñ]+", text, flags=re.IGNORECASE)
    out: list[str] = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if (
            WORD_PATTERN.fullmatch(token)
            and i + 2 < len(parts)
            and parts[i + 1].isspace()
            and WORD_PATTERN.fullmatch(parts[i + 2])
        ):
            left = token
            sep = parts[i + 1]
            right = parts[i + 2]
            left_lower = left.lower()
            right_lower = right.lower()
            right_common = is_common_word(right_lower)
            next_is_word = (
                i + 4 < len(parts)
                and parts[i + 3].isspace()
                and WORD_PATTERN.fullmatch(parts[i + 4])
            )

            if left_lower == "est" and len(right_lower) == 1 and right_lower in VOWELS:
                out.append(preserve_case("está", left + right))
                i += 3
                continue

            if left_lower == "est" and right_common:
                replacement = "está"
                if right_lower not in ESTA_VERB_NEXT and not right_lower.endswith(("ando", "iendo", "yendo")):
                    replacement = "esta"
                out.append(f"{preserve_case(replacement, left)}{sep}{right}")
                i += 3
                continue

            if (
                len(right_lower) == 1
                and next_is_word
                and len(left_lower) > 1
                and is_common_word(left_lower)
            ):
                next_word = parts[i + 4]
                next_lower = next_word.lower()
                right_join, right_score, _ = best_vowel_join(right_lower, next_lower)
                if right_join and right_score >= MIN_ZIPF_JOIN_STRICT:
                    out.append(token)
                    i += 1
                    continue

            join_word, join_score, second_score = best_vowel_join(left_lower, right_lower)
            join_gap = join_score - second_score

            if should_merge_parts(left_lower, right_lower) and join_word:
                if join_score >= MIN_ZIPF_JOIN_STRICT:
                    out.append(preserve_case(join_word, left + right))
                    i += 3
                    continue
                if join_score >= MIN_ZIPF_JOIN_LOOSE and join_gap >= MIN_ZIPF_JOIN_GAP:
                    out.append(preserve_case(join_word, left + right))
                    i += 3
                    continue

        out.append(token)
        i += 1

    return "".join(out)


def expand_chat_tokens(text: str) -> str:
    for short, long_form in CHAT_EXPANSIONS.items():
        pattern = rf"\b{re.escape(short)}\b"
        def repl(match, short=short, long_form=long_form):
            original = match.group(0)
            if short == "x":
                prefix = text[:match.start()]
                prev = re.search(r"([a-záéíóúüñ]+)\s*$", prefix, flags=re.IGNORECASE)
                if prev and prev.group(1).lower() == "muy":
                    return original
            if original.isupper() and len(original) <= 2:
                prefix = text[:match.start()]
                if match.start() == 0 or re.search(r"[.!?]\s*$", prefix):
                    return long_form.capitalize()
                return long_form
            return preserve_case(long_form, original)

        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def apply_compr_context(text: str) -> str:
    noun_prefixes = (
        "mi|tu|su|mis|tus|sus|nuestra|nuestro|nuestras|nuestros|"
        "una|un|unos|unas|"
        "excelente|buena|buen|gran|mala|malo|perfecta|perfecto|"
        "bonita|bonito|linda|lindo|mejor|peor|nueva|nuevo"
    )
    noun_pattern = rf"\b(?P<prefix>{noun_prefixes})\s+(?P<compr>compr)\b"
    text = re.sub(
        noun_pattern,
        lambda m: f"{m.group('prefix')} {preserve_case('compra', m.group('compr'))}",
        text,
        flags=re.IGNORECASE,
    )

    clitics = "me|te|se|nos|les"
    objects = "la|lo|las|los"
    clitic_obj_pattern = rf"\b(?P<clitic>{clitics})\s+(?P<obj>{objects})\s+(?P<compr>compr)\b"
    text = re.sub(
        clitic_obj_pattern,
        lambda m: f"{m.group('clitic')} {m.group('obj')} {preserve_case('compré', m.group('compr'))}",
        text,
        flags=re.IGNORECASE,
    )

    obj_pattern = rf"\b(?P<obj>{objects})\s+(?P<compr>compr)\b"
    text = re.sub(
        obj_pattern,
        lambda m: f"{m.group('obj')} {preserve_case('compré', m.group('compr'))}",
        text,
        flags=re.IGNORECASE,
    )

    dets = (
        "la|el|los|las|un|una|unos|unas|"
        "este|esta|estos|estas|ese|esa|esos|esas|"
        "mi|mis|tu|tus|su|sus|nuestro|nuestra|nuestros|nuestras"
    )
    verb_pattern = rf"\b(?P<compr>compr)\s+(?P<det>{dets})\b"
    text = re.sub(
        verb_pattern,
        lambda m: f"{preserve_case('compré', m.group('compr'))} {m.group('det')}",
        text,
        flags=re.IGNORECASE,
    )

    subj_pattern = r"\b(?P<subj>yo)\s+(?P<compr>compr)\b"
    text = re.sub(
        subj_pattern,
        lambda m: f"{m.group('subj')} {preserve_case('compré', m.group('compr'))}",
        text,
        flags=re.IGNORECASE,
    )

    return text


def apply_confusables(text: str) -> str:
    text = apply_compr_context(text)
    for pattern, replacement in CONFUSABLES:
        text = re.sub(pattern, lambda m: preserve_case(replacement, m.group(0)), text, flags=re.IGNORECASE)
    return text


def should_keep_corpus_word(word: str) -> bool:
    if word in TECH_TERMS:
        return True
    return zipf_frequency(word, "es") >= MIN_ZIPF_CORPUS


def build_symspell(corpus_counts: Counter, protected_terms: set[str]) -> SymSpell:
    sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

    for word, count in corpus_counts.items():
        if should_keep_corpus_word(word):
            sym_spell.create_dictionary_entry(word, count)

    for term in protected_terms:
        sym_spell.create_dictionary_entry(term, 100000)

    for word in top_n_list("es", 50000):
        freq = max(1, int(10 ** zipf_frequency(word, "es")))
        sym_spell.create_dictionary_entry(word, freq)

    return sym_spell


def correct_token(
    token: str,
    sym_spell: SymSpell,
    cache: dict[str, str],
    protected_terms: set[str],
) -> str:
    lower = token.lower()
    if lower in cache:
        return preserve_case(cache[lower], token)

    if len(lower) <= 2:
        cache[lower] = lower
        return token
    if any(ch.isdigit() for ch in lower):
        cache[lower] = lower
        return token
    if lower in protected_terms:
        cache[lower] = lower
        return token

    token_freq = zipf_frequency(lower, "es")
    if token_freq >= MIN_ZIPF_KEEP:
        cache[lower] = lower
        return token

    if sym_spell.lookup(lower, Verbosity.TOP, max_edit_distance=0):
        cache[lower] = lower
        return token

    suggestions = sym_spell.lookup(lower, Verbosity.TOP, max_edit_distance=1)
    if suggestions:
        candidate = suggestions[0].term
        if consonant_skeleton(candidate) != consonant_skeleton(lower):
            cache[lower] = lower
            return token
        candidate_freq = zipf_frequency(candidate, "es")
        if candidate_freq >= token_freq + MIN_ZIPF_GAIN:
            cache[lower] = candidate
            return preserve_case(candidate, token)

    cache[lower] = lower
    return token


def correct_text(
    text: str,
    sym_spell: SymSpell,
    cache: dict[str, str],
    protected_terms: set[str],
) -> str:
    parts = re.split(r"([a-záéíóúüñ]+)", text, flags=re.IGNORECASE)
    out = []
    for part in parts:
        if WORD_PATTERN.fullmatch(part):
            out.append(correct_token(part, sym_spell, cache, protected_terms))
        else:
            out.append(part)
    return "".join(out)


def build_corpus_counts(df: pd.DataFrame) -> Counter:
    counts: Counter = Counter()
    for raw in df.get("raw_text", []):
        if isinstance(raw, str) and raw.strip():
            text = normalize_text(raw).lower()
            text = apply_encoding_map(text)
            text = expand_chat_tokens(text)
            text = merge_broken_words(text)
            text = apply_confusables(text)
            tokens = WORD_PATTERN.findall(text)
            counts.update(tokens)
    return counts


def main() -> None:
    df = load_csv(CSV_PATH)

    if "raw_text" not in df.columns:
        raise RuntimeError("No existe la columna raw_text para reconstruir.")

    corpus_counts = build_corpus_counts(df)
    protected_terms = set(TECH_TERMS)
    sym_spell = build_symspell(corpus_counts, protected_terms)
    cache: dict[str, str] = {}

    total_rows = len(df)
    rows_processed = 0
    rows_skipped = 0
    rows_changed = 0
    changes_log = []

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Procesando", unit="fila"):
        raw = row.get("raw_text")
        if not isinstance(raw, str) or not raw.strip():
            rows_skipped += 1
            continue

        rows_processed += 1
        original = normalize_text(raw)

        text = original
        text = apply_encoding_map(text)
        text = expand_chat_tokens(text)
        text = merge_broken_words(text)
        text = apply_confusables(text)
        text = correct_text(text, sym_spell, cache, protected_terms)
        text = normalize_text(text)

        if text != row.get("final_clean_text"):
            df.at[idx, "final_clean_text"] = text
            rows_changed += 1
            review_id = row.get("id_review") or f"row_{idx}"
            changes_log.append((review_id, original, text))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".bak.{timestamp}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    report_path = BASE_DIR / f"reporte_contextual_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Correcciones contextuales (antes y despues)\n")
        f.write("=" * 72 + "\n")
        for review_id, before, after in changes_log:
            f.write(f"ID: {review_id}\n")
            f.write(f"ANTES: {before}\n")
            f.write(f"DESPUES: {after}\n")
            f.write("-" * 72 + "\n")

    print("--- REPORTE CONTEXTUAL ---")
    print(f"Filas procesadas: {rows_processed}")
    print(f"Filas omitidas (sin texto): {rows_skipped}")
    print(f"Filas corregidas: {rows_changed}")
    print(f"Backup creado: {backup_path}")
    print(f"Reporte generado: {report_path}")


if __name__ == "__main__":
    main()
