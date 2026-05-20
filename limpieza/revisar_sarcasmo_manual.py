import csv
import random

# Leer el CSV
csv_file = 'corpus_limpio_final_fase3.csv'
rows = []
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Filtrar filas con possible_sarcasm = True y sarcasm_action_final = 'transformar'
sarcasm_rows = [row for row in rows if row.get('possible_sarcasm', '').lower() == 'true' and row.get('sarcasm_action_final', '').lower() == 'transformar']

# Seleccionar una muestra aleatoria de 10 filas (o menos si hay menos)
sample_size = min(10, len(sarcasm_rows))
sample = random.sample(sarcasm_rows, sample_size)

print(f"Total de filas con sarcasmo transformado: {len(sarcasm_rows)}")
print(f"Muestra seleccionada: {sample_size} filas")
print("\nRevisión manual de limpieza de sarcasmo (solo transformaciones):")
print("=" * 80)

for i, row in enumerate(sample, 1):
    print(f"\nFila {i}:")
    print(f"ID: {row.get('id_review', 'N/A')}")
    print(f"Texto antes de sarcasmo: {row.get('clean_text_original_pre_sarcasm', 'N/A')}")
    print(f"Texto después de sarcasmo: {row.get('text_after_sarcasm', 'N/A')}")
    print(f"Razón de sarcasmo: {row.get('sarcasm_reason', 'N/A')}")
    print(f"Notas de procesamiento: {row.get('sarcasm_processing_notes', 'N/A')}")
    print("-" * 80)