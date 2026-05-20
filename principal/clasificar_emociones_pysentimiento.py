import re
import time
import unicodedata
import hashlib
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from pysentimiento import create_analyzer
except ImportError as exc:
    raise ImportError(
        "Falta pysentimiento. Instala con: pip install pysentimiento"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
CSV_INPUT = BASE_DIR / "corpus_limpio_final_fase3_CORREGIDO.csv"
CSV_OUTPUT = BASE_DIR / "corpus_limpio_final_fase3_CORREGIDO_fase4_emociones.csv"

LABEL_MAP: dict[str, str] = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "anger": "Enojo",
    "surprise": "Sorpresa",
    "disgust": "Asco",
    "fear": "Miedo",
    "others": "Neutral",
}

POSITIVE_KEYWORDS = {
    "excelente", "bueno", "buena", "buen",  # ✅ FIX_CRITICO: "buen" añadido
    "recomendado", "recomendable",
    "recomiendo", "perfecto", "perfecta", "encantada", "encantado",
    "encanta", "encanto", "genial", "increible", "maravilloso",
    "maravillosa", "fantastico", "cumple", "cumplio", "util",
    "agradable", "satisfecho", "satisfecha", "feliz", "contento",
    "contenta", "amor", "amo",
    "funcional", "funciona", "funciono", "veloz", "rapida", "rapido",
    "ideal", "practica", "practico", "expectativas", "espectativas",
}

NEGATIVE_KEYWORDS = {
    "calienta", "caliento", "calentamiento",
    "congela", "congelo",
    "defectuoso", "defectuosa",
    "error", "fallo", "falla",
    "lag", "lento", "lenta",
    "pantallazo", "problema",
    "ruido", "ruidoso", "ruidosa",
    "rompio", "sobrecalent",
    "traba", "trabo",
    "apaga", "apago",
    "reinicio", "reinicia",
    "decepcion", "decepcionado", "decepcionada",
    "roto", "rota", "inservible", "inutil",
    "malo", "mala", "pesimo", "pesima",
    "terrible", "horrible", "basura", "mierda",
}

HARDWARE_TERMS = {
    "bateria", "pantalla", "teclado", "ventilador", "temperatura",
    "gpu", "cpu", "ram", "ssd", "disco", "cargador", "speaker",
    "bocina", "camara", "touchpad", "mouse",
}

NEGATIVE_ADJECTIVES = {
    "ruidoso", "ruidosa", "lento", "lenta", "malo", "mala",
    "defectuoso", "defectuosa", "corto", "corta", "poco",
    "insuficiente", "bajo", "baja", "debil", "fragil",
    "barato", "barata",
}

NEGATIONS = {"no", "nunca", "jamas", "cero", "sin", "ningun", "ninguna"}

SADNESS_WORDS = {
    "triste", "tristeza", "lament", "pena", "decepcion",
    "lastima", "arrepent", "frustrad", "decepcionad", "desilusionad",
}

STRONG_NEGATIVE_PATTERNS = [
    ("no", "sirve"), ("no", "funciona"), ("no", "funciono"),
    ("no", "enciende"), ("no", "encendio"), ("no", "prende"), ("no", "prendio"),
    ("no", "carga"), ("no", "cargo"), ("no", "da", "video"),
    ("no", "reconoce"), ("no", "detecta"), ("no", "recomiendo"), ("no", "recomendaria"),
    ("no", "compre"), ("no", "compraria"), ("falla", "grave"), ("fallo", "grave"),
    ("pantalla", "azul"), ("pantalla", "negra"), ("pantalla", "blanca"),
    ("pantalla", "rota"), ("pantalla", "rayada"), ("dinero", "perdido"),
    ("perdi", "mi", "dinero"), ("no", "es", "bueno"), ("no", "es", "buena"),
    ("no", "es", "perfecto"), ("no", "es", "perfecta"),
    ("no", "esta", "bueno"), ("no", "esta", "buena"),
]

NEGATABLE = {
    ("se", "congela"), ("se", "congelo"), ("se", "reinicia"), ("se", "reinicio"),
    ("se", "traba"), ("se", "trabo"), ("se", "rompe"), ("se", "rompio"),
    ("se", "apaga"), ("se", "apago"),
}

CATASTROPHIC_PREFIXES = ("explot", "incendi", "quem")

REPORTING_TERMS = {
    "dicen", "coment", "comentan", "mencion", "segun", "algunos",
    "escuche", "lei", "vi", "leyeron", "escuchado",
}

WORD_PATTERN = re.compile(r"[a-z]+")
POSITIVE_QUESTION_REGEX = re.compile(r"[¿]?\s*\b(excelente|bien|bueno)\b[,\s]*[?]+")
SURPRISE_CUE_REGEX = re.compile(
    r"(sorprend|inesper|no\s+esperab|no\s+me\s+esperab|jamas\s+imagin|increible[,:\s]+que)"
)
CONTRAST_REGEX = re.compile(
    r"\b(pero|aunque|sin\s+embargo|a\s+pesar\s+de|no\s+obstante|empero|salvo)\b"
)


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {path}")


def get_file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def get_rating_value(rating_num) -> Optional[int]:
    try:
        return int(float(rating_num))
    except (TypeError, ValueError):
        return None


def is_negated(index: int, negation_indexes: list[int]) -> bool:
    return any(0 < (index - neg_index) <= 3 for neg_index in negation_indexes)


def extract_focus_text(text: str) -> str:
    matches = list(CONTRAST_REGEX.finditer(text))
    if not matches:
        return text
    if len(matches) == 1:
        tail = text[matches[0].end():].strip()
        before = text[:matches[0].start()].strip()
        return tail if len(tail) >= len(before) else before

    # ✅ CORRECCIÓN: Manejo inteligente cuando hay 2+ contrastes
    last = matches[-1]
    tail = text[last.end():].strip()

    if tail:
        return tail

    # Si no hay nada después del último contraste, tomar lo que hay ANTES del primero
    first = matches[0]
    before = text[:first.start()].strip()

    return before if before else text


def has_reporting_context(tokens: list[str], index: int) -> bool:
    start = max(index - 4, 0)
    return any(tok in REPORTING_TERMS for tok in tokens[start:index])


def has_negation_after(tokens: list[str], index: int, window: int = 6) -> bool:
    end = min(index + window + 1, len(tokens))
    return any(tok in NEGATIONS for tok in tokens[index + 1:end])


def has_strong_negative_event(tokens: list[str]) -> bool:
    negation_indexes = [i for i, tok in enumerate(tokens) if tok in NEGATIONS]

    for pattern in STRONG_NEGATIVE_PATTERNS:
        plen = len(pattern)
        for i in range(len(tokens) - plen + 1):
            if tuple(tokens[i:i + plen]) == pattern:
                # Los patrones que ya contienen "no" son explícitamente negativos
                # Los patrones sin "no" deben verificarse contra negación externa
                if "no" not in pattern:
                    if is_negated(i, negation_indexes):
                        continue
                return True

    for pattern in NEGATABLE:
        plen = len(pattern)
        for i in range(len(tokens) - plen + 1):
            if tuple(tokens[i:i + plen]) == pattern:
                if is_negated(i, negation_indexes) or has_negation_after(tokens, i):
                    continue
                return True
    return False


def has_catastrophic_event(tokens: list[str]) -> bool:
    negation_indexes = [i for i, tok in enumerate(tokens) if tok in NEGATIONS]
    for i, tok in enumerate(tokens):
        if tok in {"fuego", "humo", "chispa"}:
            if is_negated(i, negation_indexes) or has_reporting_context(tokens, i):
                continue
            return True
        if tok.startswith(CATASTROPHIC_PREFIXES):
            if is_negated(i, negation_indexes) or has_reporting_context(tokens, i):
                continue
            if has_negation_after(tokens, i):
                continue
            return True
    return False


def count_negative_signals(tokens: list[str]) -> Tuple[int, int]:
    negation_indexes = [i for i, tok in enumerate(tokens) if tok in NEGATIONS]
    negative_count = 0
    positive_from_negation = 0
    counted_indexes = set()

    for i, tok in enumerate(tokens):
        if tok in NEGATIVE_KEYWORDS:
            counted_indexes.add(i)
            if is_negated(i, negation_indexes):
                positive_from_negation += 1
            else:
                negative_count += 1

    for i, tok in enumerate(tokens):
        if tok in HARDWARE_TERMS:
            window_start = max(i - 3, 0)
            window_end = min(i + 4, len(tokens))
            for j in range(window_start, window_end):
                if j in counted_indexes:
                    continue
                adj = tokens[j]
                if adj in NEGATIVE_ADJECTIVES and not is_negated(j, negation_indexes):
                    negative_count += 1
                    counted_indexes.add(j)
                    break
    return negative_count, positive_from_negation


def count_positive_signals(tokens: list[str]) -> int:
    negation_indexes = [i for i, tok in enumerate(tokens) if tok in NEGATIONS]
    count = 0
    for i, tok in enumerate(tokens):
        if tok in POSITIVE_KEYWORDS:
            if not is_negated(i, negation_indexes):
                count += 1
    return count


def corregir_emocion_ecommerce(texto_limpio: str, emocion_base: str, rating_num) -> str:
    if not isinstance(texto_limpio, str):
        return ""
    normalized = normalize_text(texto_limpio)
    if not normalized or normalized.strip() in {"nan", "none", "null"}:
        return ""

    focus_text = extract_focus_text(normalized)
    tokens = tokenize(focus_text)
    tokens_set = set(tokens)

    rating_value = get_rating_value(rating_num)
    is_low_rating = rating_value in (1, 2)
    is_mid_rating = rating_value == 3
    is_high_rating = rating_value in (4, 5)

    # ✅ CORRECCIÓN: Cálculo único de señales (antes se calculaban 2 veces)
    positive_count = count_positive_signals(tokens)
    negative_count, positive_from_negation = count_negative_signals(tokens)
    total_positive = positive_count + positive_from_negation # Usar esta suma de aquí en adelante

    has_strong_negative = has_strong_negative_event(tokens)
    has_catastrophic = has_catastrophic_event(tokens)
    has_sadness = bool(tokens_set.intersection(SADNESS_WORDS))
    has_contrast = bool(CONTRAST_REGEX.search(normalized))

    if is_high_rating:
        if has_catastrophic: return "Enojo"
        if has_strong_negative: return "Neutral"
        if negative_count >= 2 or (negative_count >= 1 and has_contrast): return "Neutral"
        if total_positive > 0: return "Alegría"
        return emocion_base

    if is_low_rating:
        if has_catastrophic: return "Enojo"
        if has_strong_negative or negative_count > 0:
            return "Tristeza" if has_sadness else "Enojo"
        return emocion_base if emocion_base in ("Enojo", "Tristeza") else "Neutral"

    if is_mid_rating:
        if total_positive > negative_count: return "Alegría"
        if has_strong_negative or negative_count > total_positive: return "Neutral"
        if emocion_base in ("Sorpresa", "Miedo"): return emocion_base
        return "Neutral"

    if emocion_base == "Neutral":
        if has_catastrophic or (has_strong_negative and negative_count > 0): return "Enojo"
        if total_positive > 0: return "Alegría"

    if emocion_base in ("Tristeza", "Miedo"):
        if negative_count > 0: return "Enojo"
        # ✅ CORRECCIÓN: Eliminada la condición muerta "is_high_rating"
        # y se usa "total_positive" en lugar de recalcular
        if total_positive > 0 and has_contrast: return "Alegría"

    if emocion_base == "Alegría":
        if has_strong_negative: return "Enojo"
        if negative_count > total_positive: return "Neutral"

    if emocion_base == "Sorpresa":
        if POSITIVE_QUESTION_REGEX.search(normalized): return "Alegría"
        if not SURPRISE_CUE_REGEX.search(normalized) and tokens_set.intersection(POSITIVE_KEYWORDS):
            return "Alegría"

    if emocion_base == "Asco":
        if has_strong_negative: return "Enojo"

    return emocion_base


def main() -> None:
    print("=" * 60)
    print("FASE 4a: CLASIFICACIÓN DE EMOCIONES")
    print("=" * 60)

    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada:\n  {CSV_INPUT}")

    input_hash = get_file_hash(CSV_INPUT)
    print(f"Archivo: {CSV_INPUT.name} | Hash: {input_hash} | {time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_csv(CSV_INPUT)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna 'final_clean_text' en el CSV.")
    if "emotion" in df.columns:
        print("⚠️  La columna 'emotion' ya existe. Se sobreescribirá.")
        if input("   ¿Continuar? (s/n): ").strip().lower() != "s":
            return

    df["final_clean_text"] = df["final_clean_text"].fillna("")

    print("\nCargando modelo pysentimiento...")
    analyzer = create_analyzer(task="emotion", lang="es")
    print("Modelo listo.\n")

    emotions = []
    stats = {"total": len(df), "vacios": 0, "corregidos": 0, "sin_cambio": 0}

    for row in tqdm(df.itertuples(), total=len(df), desc="Clasificando", unit="fila"):
        text = getattr(row, "final_clean_text", None)
        rating_num = getattr(row, "rating_num", None)

        if not isinstance(text, str) or not text.strip():
            emotions.append("")
            stats["vacios"] += 1
            continue

        result = analyzer.predict(text)
        base_emotion = LABEL_MAP.get(result.output, "Neutral")
        corrected = corregir_emocion_ecommerce(text, base_emotion, rating_num)

        if corrected != base_emotion:
            stats["corregidos"] += 1
        else:
            stats["sin_cambio"] += 1
        emotions.append(corrected)

    df["emotion"] = emotions
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    output_hash = get_file_hash(CSV_OUTPUT)
    valid = max(1, stats["total"] - stats["vacios"])

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Procesadas: {stats['total']} | Vacías: {stats['vacios']}")
    print(f"Corregidas: {stats['corregidos']} ({stats['corregidos'] / valid * 100:.1f}%)")
    print(f"Sin cambio: {stats['sin_cambio']}")
    print(f"Entrada: {CSV_INPUT.name} ({input_hash})")
    print(f"Salida:  {CSV_OUTPUT.name} ({output_hash})")

    print(f"\n--- EMOCIONES ---")
    t = max(1, len(df))
    for emo, cnt in df["emotion"].value_counts().items():
        print(f"  {emo:12s}: {cnt:6d} ({cnt / t * 100:5.1f}%)")


if __name__ == "__main__":
    main()