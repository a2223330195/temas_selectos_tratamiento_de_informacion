import pandas as pd
import time
import os
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. CONFIGURACIÓN (CLAVE INSERTADA)
# ==========================================
# NOTA: Te sugiero reemplazar esta clave por una nueva después de borrar la anterior
GROQ_API_KEY = "quitada por seguridad"  # Reemplaza con tu clave de API de Groq (OpenAI)
INPUT_FILE = "raw_laptops_ml.csv"
OUTPUT_FILE = "corpus_limpio_ml.csv"
COLUMNA_ORIGINAL = "raw_text"
COLUMNA_LIMPIA = "clean_text"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
Eres un experto en lingüística y normalización de corpus de texto para NLP.
Tu tarea es recibir un comentario de un usuario sobre un producto (en español) y devolverlo **completamente limpio** siguiendo estas reglas estrictas:
1. Ortografía: Corrige faltas (tildes, b/v, s/c/z, h, g/j).
2. Semántica y Puntuación: Corrige la puntuación y elimina palabras de más.
3. Modismos y Jerga: Reemplaza modismos o jerga de internet (ej. "jala chido", "la neta") por un español neutro y claro.
4. Limpieza visual: Elimina TODOS los emojis y emoticones.
5. Formato: Devuelve ÚNICAMENTE el texto corregido.
"""


# ==========================================
# 2. FUNCIÓN DE LIMPIEZA CON REINTENTOS
# ==========================================
def limpiar_comentario(texto_original):
    if pd.isna(texto_original) or str(texto_original).strip() == "":
        return ""

    intentos = 0
    max_intentos = 3

    while intentos < max_intentos:
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Corrige el siguiente comentario: {texto_original}"}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            intentos += 1
            error_msg = str(e)

            if "401" in error_msg or "Invalid API Key" in error_msg:
                print("\n🛑 CLAVE INVÁLIDA. Deteniendo script.")
                return "ERROR_CLAVE_INVALIDA"

            time.sleep(10)

    return str(texto_original)


# ==========================================
# 3. PROCESAMIENTO PRINCIPAL
# ==========================================
def procesar_corpus():
    print(f"📂 Cargando archivo: {INPUT_FILE}...")

    if os.path.exists(OUTPUT_FILE):
        print("🔄 Se encontró un archivo de salida previo. Cargando progreso...")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        try:
            df = pd.read_csv(INPUT_FILE)
        except FileNotFoundError:
            print(f"Error: No se encontró {INPUT_FILE}")
            return

    if COLUMNA_ORIGINAL not in df.columns:
        print(f"Error: La columna '{COLUMNA_ORIGINAL}' no existe.")
        return

    # Solución al error float64: Forzar a que sea tipo texto
    if COLUMNA_LIMPIA not in df.columns:
        df[COLUMNA_LIMPIA] = ""
    else:
        df[COLUMNA_LIMPIA] = df[COLUMNA_LIMPIA].astype('object').fillna("")

    total_filas = len(df)
    print(f"✅ Total de comentarios en el archivo: {total_filas}")
    print("🚀 Iniciando limpieza (Esto tomará un par de horas)...\n")

    barra_progreso = tqdm(df.iterrows(), total=total_filas, desc="Limpiando texto")

    for index, row in barra_progreso:
        texto_raw = row[COLUMNA_ORIGINAL]
        texto_actual_limpio = row[COLUMNA_LIMPIA]

        # Si ya está limpio (por si se reinició la pc), lo salta
        if pd.notna(texto_actual_limpio) and str(texto_actual_limpio).strip() != "":
            continue

        resultado = limpiar_comentario(texto_raw)

        if resultado == "ERROR_CLAVE_INVALIDA":
            print("\n\n❌ Script detenido. Arregla tu API Key y vuelve a correr.")
            return

        df.at[index, COLUMNA_LIMPIA] = resultado

        # Guardado automático cada 100 comentarios
        if (index + 1) % 100 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
            barra_progreso.set_postfix_str(f"💾 Guardado en {OUTPUT_FILE}")

        time.sleep(1.5)  # Pausa para no saturar la API

    # Guardado Final
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n\n🎉 ¡PROCESO 100% TERMINADO Y GUARDADO CON ÉXITO!")
    print(f"📁 Puedes encontrar tu corpus limpio aquí: {OUTPUT_FILE}")


if __name__ == "__main__":
    procesar_corpus()