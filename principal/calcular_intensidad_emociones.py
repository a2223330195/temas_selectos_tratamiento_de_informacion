# calcular_intensidad_emociones.py
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

BASE_DIR = Path(__file__).resolve().parent
CSV_INPUT = BASE_DIR / "corpus_limpio_final_fase3_CORREGIDO_fase4_emociones.csv"
CSV_OUTPUT = BASE_DIR / "corpus_limpio_final_fase3_CORREGIDO_fase4_completo.csv"

MITIGATORS = {
    "un poco", "algo", "ligeramente", "casi", "medio",
    "mas o menos", "regular", "normalito", "normalita",
}

DOUBT_PHRASES = {
    "creo que", "parece que", "tal vez", "quizas",
    "no se", "me parece", "supongo",
}

CONTRAST_REGEX = re.compile(
    r"\b(pero|aunque|sin\s+embargo|a\s+pesar\s+de|no\s+obstante|empero|salvo)\b"
)

INTENSIFIERS = {
    "muy", "super", "bastante", "demasiado", "altamente", "sumamente",
    "extremadamente", "realmente", "totalmente", "absolutamente",
    "completamente", "definitivamente", "mucho",
}

ABSOLUTES = {
    "excelente", "pesimo", "horrible", "maravilloso", "maravillosa",
    "encanta", "encanto", "odio", "perfecto", "perfecta",
    "increible", "fantastico", "genial", "terrorifico", "espantoso", "espantosa",
}

DETERMINERS = {"nunca", "siempre", "jamas", "nada", "todo", "todos", "ningun", "ninguna"}

POSITIVE_WORDS = {
    "bueno", "buena", "excelente", "recomendado", "recomendable",
    "encanta", "encanto", "perfecto", "perfecta", "increible",
    "cumple", "util", "genial", "fantastico", "maravilloso", "maravillosa",
}

CATASTROPHIC_WORDS = {"fuego", "humo", "chispa"}
CATASTROPHIC_PREFIXES = {"explot", "incendi", "quem"}

LEGAL_TERMS = {
    "profeco", "demanda", "abogado", "fraude", "estafa", "robo",
    "denuncia", "fiscalia", "juicio", "querella"
}

FINANCIAL_TERMS = {
    "reembolso", "devolucion", "quiero mi dinero", "exijo reembolso",
    "reembolsen", "devuelvanme", "dinero perdido"
}

EXTREME_LOYALTY = {
    "me cambia la vida", "mejor producto de la historia",
    "compraria mil", "compraria mil mas", "compraria 1000",
    "la mejor compra", "el mejor producto", "nunca mejor inversion",
}

PROFANITY_EXACT = {
    "mierda", "basura", "idiota", "pendejo", "estupido", "pinche",
    "verga", "culero", "huevon", "puta", "puto"
}
PROFANITY_PREFIXES = {"ching"}

WORD_PATTERN = re.compile(r"[a-z]+")
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
MULTI_PUNCT_PATTERN = re.compile(r"([!?])\1{1,}")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")

SATURATION_THRESHOLD = 4

REPORTING_TERMS = {
    "dicen", "comentan", "coment", "mencion", "segun", "algunos",
    "escuche", "lei", "vi", "leyeron", "escuchado",
}

# ==============================================================================
# FUNCIONES GLOBALES EXTRADAS (Principio DRY y Lógica Unidireccional)
# ==============================================================================

def is_negated(index: int, negation_indexes: list[int]) -> bool:
    return any(0 < (index - neg_index) <= 3 for neg_index in negation_indexes)


def has_reporting_context(tokens: list[str], index: int) -> bool:
    """Verifica si el evento catastrófico está en un contexto de reporte de terceros."""
    start = max(index - 4, 0)
    return any(tok in REPORTING_TERMS for tok in tokens[start:index])

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

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


def get_rating_value(rating_num, rating_text=None) -> Optional[int]:
    try:
        return int(float(rating_num))
    except (TypeError, ValueError):
        pass
    if isinstance(rating_text, str):
        match = re.search(r"(\d+)", rating_text)
        if match:
            return int(match.group(1))
    return None


def has_any_phrase(text: str, phrases: set[str]) -> bool:
    for phrase in phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return True
    return False

# ==============================================================================
# LÓGICA PRINCIPAL DE INTENSIDAD
# ==============================================================================

def score_intensity(text: str, norm_text: str, tokens: list[str]) -> Tuple[int, bool, int, bool]:
    negation_indexes = [i for i, tok in enumerate(tokens) if
                        tok in {"no", "nunca", "jamas", "cero", "sin", "ningun", "ninguna"}]

    score = 0
    if has_any_phrase(norm_text, MITIGATORS): score -= 1
    if has_any_phrase(norm_text, DOUBT_PHRASES): score -= 1

    for i, tok in enumerate(tokens):
        if tok in INTENSIFIERS and not is_negated(i, negation_indexes): score += 1
    for i, tok in enumerate(tokens):
        if tok in ABSOLUTES and not is_negated(i, negation_indexes): score += 1
    for i, tok in enumerate(tokens):
        if tok in DETERMINERS and not is_negated(i, negation_indexes): score += 1

    has_upper = bool(UPPER_WORD_PATTERN.search(text))
    has_multi_punct = bool(MULTI_PUNCT_PATTERN.search(text))
    has_elongation = bool(ELONGATION_PATTERN.search(text))

    if has_upper: score += 1
    if has_multi_punct: score += 1
    if has_elongation: score += 1

    has_contrast = bool(CONTRAST_REGEX.search(norm_text))
    if has_contrast:
        for i, tok in enumerate(tokens):
            if tok in POSITIVE_WORDS and not is_negated(i, negation_indexes):
                score -= 1
                break

    intensifier_count = sum(1 for i, tok in enumerate(tokens) if tok in INTENSIFIERS and not is_negated(i, negation_indexes))

    saturation = bool(
        intensifier_count >= 2 and
        (has_upper or has_multi_punct or has_elongation) and
        ((has_upper and has_multi_punct) or has_elongation)
    )

    return score, saturation, intensifier_count, has_contrast


def has_override_level_4(text: str, norm_text: str, tokens: list[str]) -> bool:
    negation_indexes = [i for i, tok in enumerate(tokens) if
                        tok in {"no", "nunca", "jamas", "cero", "sin", "ningun", "ninguna"}]

    for i, tok in enumerate(tokens):
        if tok in CATASTROPHIC_WORDS:
            if is_negated(i, negation_indexes) or has_reporting_context(tokens, i): continue
            return True
        if any(tok.startswith(p) for p in CATASTROPHIC_PREFIXES):
            if is_negated(i, negation_indexes) or has_reporting_context(tokens, i): continue
            return True

    for phrase in LEGAL_TERMS | FINANCIAL_TERMS:
        phrase_tokens = phrase.split()
        plen = len(phrase_tokens)
        for i in range(len(tokens) - plen + 1):
            if tokens[i:i + plen] == phrase_tokens:
                # CORRECCIÓN: Verifica si CUALQUIER palabra de la frase completa está negada
                if not any(is_negated(j, negation_indexes) for j in range(i, i + plen)):
                    return True
                break

    for phrase in EXTREME_LOYALTY:
        phrase_tokens = phrase.split()
        plen = len(phrase_tokens)
        for i in range(len(tokens) - plen + 1):
            if tokens[i:i + plen] == phrase_tokens:
                # CORRECCIÓN: Verifica si CUALQUIER palabra de la frase completa está negada
                if not any(is_negated(j, negation_indexes) for j in range(i, i + plen)):
                    return True
                break

    for i, tok in enumerate(tokens):
        if tok in PROFANITY_EXACT or any(tok.startswith(p) for p in PROFANITY_PREFIXES):
            if not is_negated(i, negation_indexes): return True

    return False


def compute_intensity(text: str, norm_text: str, tokens: list[str], rating_num, rating_text=None, emotion=None) -> Tuple[int, bool, bool, int]:
    if not isinstance(text, str) or not text.strip():
        return 2, False, False, 0

    override = has_override_level_4(text, norm_text, tokens)
    score, saturation, _, has_contrast = score_intensity(text, norm_text, tokens)

    # 1. Nivel Base
    if override or saturation or score >= SATURATION_THRESHOLD:
        intensity = 4
    elif score < 0:
        intensity = 1
    elif score == 0:
        intensity = 2
    else:
        intensity = 3

    # 2. Ajuste por Rating
    rating_value = get_rating_value(rating_num, rating_text)
    if rating_value == 3 and intensity == 4 and not override:
        intensity = 3 if score > 0 else 2
    if rating_value is not None:
        if rating_value in (4, 5) and intensity == 1: intensity = 2
        if rating_value == 1 and intensity == 3 and score < 2: intensity = 2

    # 3. Ajuste por Emoción
    if isinstance(emotion, str) and emotion.strip():
        emotion = emotion.strip()
        if emotion == "Enojo" and intensity < 3 and score > 0: intensity = 3
        if emotion == "Asco" and intensity < 3: intensity = 3
        if emotion == "Sorpresa" and intensity == 2 and score > 0: intensity = 3

    return intensity, override, saturation, score


def detect_incongruence(emotion: str, intensity: int, rating_value: Optional[int], norm_text: str, tokens: list[str], saturation: bool, score: int) -> Tuple[bool, str]:
    if not isinstance(emotion, str) or not emotion.strip():
        return False, ""

    emotion = emotion.strip()
    reasons = []
    tokens_set = set(tokens)

    if rating_value is not None:
        if emotion == "Alegría" and rating_value <= 2:
            reasons.append("alegria_rating_bajo")
        if emotion in ("Enojo", "Tristeza", "Asco", "Miedo") and rating_value == 5:
            reasons.append(f"{emotion.lower()}_rating_alto")

        if emotion == "Neutral" and rating_value in (1, 5):
            if has_any_phrase(norm_text, POSITIVE_WORDS) and rating_value == 1:
                reasons.append("neutral_positivo_rating_bajo")
            if has_any_phrase(norm_text, {"malo", "mala", "falla", "problema"}) and rating_value == 5:
                reasons.append("neutral_negativo_rating_alto")

        if intensity == 1 and rating_value == 5:
            reasons.append("intensidad_baja_rating_alto")
        if intensity == 4 and rating_value <= 2:
            reasons.append("intensidad_alta_rating_bajo")

    if emotion == "Enojo" and intensity in (1, 2):
        if not has_any_phrase(norm_text, MITIGATORS) and not has_any_phrase(norm_text, DOUBT_PHRASES):
            reasons.append("enojo_intensidad_baja")

    if emotion == "Alegría" and intensity == 4:
        if not has_any_phrase(norm_text, EXTREME_LOYALTY) and score < SATURATION_THRESHOLD:
            reasons.append("alegria_intensidad_extrema")

    return len(reasons) > 0, "; ".join(reasons)


def main() -> None:
    print("=" * 60)
    print("FASE 4b: CÁLCULO DE INTENSIDAD EMOCIONAL")
    print("=" * 60)

    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada:\n  {CSV_INPUT}")

    input_hash = get_file_hash(CSV_INPUT)
    print(f"Archivo: {CSV_INPUT.name} | Hash: {input_hash} | {time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_csv(CSV_INPUT)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna 'final_clean_text'.")
    if "emotion" not in df.columns:
        raise RuntimeError("No existe la columna 'emotion'. Ejecuta primero el script de emociones.")

    if "intensity" in df.columns:
        print("⚠️  La columna 'intensity' ya existe. Se sobreescribirá.")
        if input("   ¿Continuar? (s/n): ").strip().lower() != "s":
            return

    df["final_clean_text"] = df["final_clean_text"].fillna("")

    intensities = []
    incongruences = []
    incongruence_reasons = []
    stats = {"total": len(df), "i1": 0, "i2": 0, "i3": 0, "i4": 0, "incong": 0, "overrides": 0}

    for row in tqdm(df.itertuples(), total=len(df), desc="Intensidad", unit="fila"):
        text = getattr(row, "final_clean_text", "")
        rating_num = getattr(row, "rating_num", None)
        rating_text = getattr(row, "rating", None)
        emotion = getattr(row, "emotion", None)

        norm_text = normalize_text(text) if isinstance(text, str) else ""
        tokens = tokenize(norm_text)

        intensity, was_override, saturation, score = compute_intensity(
            text, norm_text, tokens, rating_num, rating_text, emotion
        )
        rating_value = get_rating_value(rating_num, rating_text)

        has_incongruencia, reason = detect_incongruence(
            emotion, intensity, rating_value, norm_text, tokens, saturation, score
        )

        stats[f"i{intensity}"] += 1
        if was_override: stats["overrides"] += 1
        if has_incongruencia: stats["incong"] += 1

        intensities.append(intensity)
        incongruences.append(has_incongruencia)
        incongruence_reasons.append(reason)

    df["intensity"] = intensities
    df["flag_incongruencia"] = incongruences
    df["incongruencia_motivo"] = incongruence_reasons

    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    output_hash = get_file_hash(CSV_OUTPUT)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    t = max(1, stats["total"])
    print(f"Procesadas: {stats['total']} | Overrides: {stats['overrides']} | Incongruencias: {stats['incong']}")
    print(f"Entrada: {CSV_INPUT.name} ({input_hash})")
    print(f"Salida:  {CSV_OUTPUT.name} ({output_hash})")
    print(f"\n--- INTENSIDADES ---")
    for lvl, key in [(1, "i1"), (2, "i2"), (3, "i3"), (4, "i4")]:
        print(f"  Nivel {lvl}: {stats[key]:6d} ({stats[key] / t * 100:5.1f}%)")
    print(f"\n--- EMOCIONES ---")
    for emo, cnt in df["emotion"].value_counts().items():
        print(f"  {emo:12s}: {cnt:6d} ({cnt / t * 100:5.1f}%)")

    if stats["incong"] > 0:
        print(f"\n--- TOP 10 INCONGRUENCIAS ---")
        inc_df = df[df["flag_incongruencia"] == True][
            ["id_review", "rating_num", "emotion", "intensity", "incongruencia_motivo", "final_clean_text"]
        ].head(10)
        for _, r in inc_df.iterrows():
            txt = str(r["final_clean_text"])[:60] + "..." if len(str(r["final_clean_text"])) > 60 else str(r["final_clean_text"])
            print(f"  {r['id_review']}: R={r['rating_num']} E={r['emotion']} I={r['intensity']}")
            print(f"    → {r['incongruencia_motivo']} | {txt}")


if __name__ == "__main__":
    main()