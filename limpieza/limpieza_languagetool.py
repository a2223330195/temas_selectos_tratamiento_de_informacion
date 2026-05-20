import re
import shutil
import time
from pathlib import Path

import pandas as pd

try:
    import language_tool_python
except ImportError as exc:
    raise SystemExit(
        "Falta language-tool-python. Instala con: pip install language-tool-python"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "corpus_limpio_final_fase3.csv"

SAFE_ISSUE_TYPES = {"whitespace", "typographical", "punctuation", "capitalization"}


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


def alnum_signature(text: str) -> str:
    return "".join(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text)).lower()


def is_safe_replacement(original: str, replacement: str) -> bool:
    return alnum_signature(original) == alnum_signature(replacement)


def apply_safe_corrections(text: str, tool) -> tuple[str, int, int, int]:
    matches = tool.check(text)
    if not matches:
        return text, 0, 0, 0

    updated = text
    applied = 0
    skipped = 0

    for match in sorted(matches, key=lambda m: m.offset, reverse=True):
        if match.rule_issue_type not in SAFE_ISSUE_TYPES:
            skipped += 1
            continue
        if not match.replacements:
            skipped += 1
            continue

        replacement = match.replacements[0]
        start = match.offset
        end = match.offset + match.error_length
        original = updated[start:end]

        if not is_safe_replacement(original, replacement):
            skipped += 1
            continue

        updated = updated[:start] + replacement + updated[end:]
        applied += 1

    return updated, applied, skipped, len(matches)


def main() -> None:
    df = load_csv(CSV_PATH)

    if "final_clean_text" not in df.columns:
        raise RuntimeError("No existe la columna final_clean_text.")

    tool = language_tool_python.LanguageTool("es")

    total_rows = len(df)
    rows_processed = 0
    rows_skipped = 0
    rows_changed = 0
    changes_log = []
    total_matches = 0
    total_applied = 0
    total_skipped = 0

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Procesando", unit="fila"):
        text = row.get("final_clean_text")
        if not isinstance(text, str) or not text.strip():
            rows_skipped += 1
            continue

        rows_processed += 1
        original = text

        corrected, applied, skipped, match_count = apply_safe_corrections(original, tool)
        corrected = normalize_text(corrected)

        total_matches += match_count
        total_applied += applied
        total_skipped += skipped

        if corrected != original:
            df.at[idx, "final_clean_text"] = corrected
            rows_changed += 1
            review_id = row.get("id_review") or f"row_{idx}"
            changes_log.append((review_id, original, corrected))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".bak.{timestamp}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    report_path = BASE_DIR / f"reporte_languagetool_safe_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Correcciones LanguageTool (antes y despues)\n")
        f.write("=" * 72 + "\n")
        for review_id, before, after in changes_log:
            f.write(f"ID: {review_id}\n")
            f.write(f"ANTES: {before}\n")
            f.write(f"DESPUES: {after}\n")
            f.write("-" * 72 + "\n")

    print("--- REPORTE Languagetool ---")
    print(f"Filas procesadas: {rows_processed}")
    print(f"Filas omitidas (sin texto): {rows_skipped}")
    print(f"Filas corregidas: {rows_changed}")
    print(f"Sugerencias revisadas: {total_matches}")
    print(f"Sugerencias aplicadas: {total_applied}")
    print(f"Sugerencias omitidas: {total_skipped}")
    print(f"Backup creado: {backup_path}")
    print(f"Reporte generado: {report_path}")


if __name__ == "__main__":
    main()
