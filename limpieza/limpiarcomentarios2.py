import pandas as pd
import time
import os
import json
import logging
import google.generativeai as genai
from tqdm import tqdm

# ==========================================
# 1. CONFIGURACIÓN Y LOGGING
# ==========================================
GEMINI_API_KEY = "quitada por seguridad"  # Reemplaza con tu clave de API de Google Gemini
INPUT_FILE = "raw_laptops_ml.csv"
OUTPUT_FILE = "corpus_limpio_ml2.csv"
COLUMNA_ORIGINAL = "raw_text"
COLUMNA_LIMPIA = "clean_text"

# Configuración de logs para auditoría nocturna
logging.basicConfig(
    filename='limpieza_corpus.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

genai.configure(api_key=GEMINI_API_KEY)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "respuestas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "texto_limpio": {"type": "string"}
                },
                "required": ["id", "texto_limpio"]
            }
        }
    },
    "required": ["respuestas"]
}

# Configuración del Modelo
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Eres un normalizador de texto. Corrige ortografía, quita jerga y traduce emojis a texto neutro. Mantén IDs."
)


# ==========================================
# 2. FUNCIONES DE ALTA DISPONIBILIDAD
# ==========================================

def llamar_api(lote_objetos, es_reintento_unitario=False):
    """Llamada base a la API con manejo de esquema."""
    try:
        payload = json.dumps({"lote": lote_objetos}, ensure_ascii=False)
        response = model.generate_content(
            payload,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": RESPONSE_SCHEMA
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        )

        # Validación de respuesta vacía (bloqueo de seguridad de Google)
        if not response.candidates or not response.candidates[0].content.parts:
            return None

        res_dict = json.loads(response.text)
        return {r['id']: r['texto_limpio'] for r in res_dict.get("respuestas", [])}

    except Exception as e:
        if not es_reintento_unitario:
            logging.warning(f"Fallo en lote: {e}")
        return None


def procesar_lote_con_fallback(lote_objetos):
    """
    Si un lote falla (posiblemente por un comentario que dispara filtros de seguridad),
    se procesa cada elemento individualmente para no perder todo el lote.
    """
    resultados = llamar_api(lote_objetos)

    if resultados:
        return resultados

    # --- INICIO DEL FALLBACK UNITARIO ---
    logging.info(f"Lote bloqueado o fallido. Iniciando procesamiento unitario para {len(lote_objetos)} items...")
    resultados_unitarios = {}

    for item in lote_objetos:
        res = llamar_api([item], es_reintento_unitario=True)
        if res:
            resultados_unitarios.update(res)
            time.sleep(1.1)  # Pausa mínima para no saturar en modo unitario
        else:
            resultados_unitarios[item['id']] = "[BLOQUEADO_POR_SEGURIDAD]"
            logging.error(f"Item ID {item['id']} bloqueado individualmente.")

    return resultados_unitarios


# ==========================================
# 3. PIPELINE DE DATOS
# ==========================================

def procesar_corpus():
    # Carga de datos
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
    else:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
        df[COLUMNA_LIMPIA] = None

    # Filtrar pendientes (Lógica robusta para evitar nulos de pandas)
    df[COLUMNA_LIMPIA] = df[COLUMNA_LIMPIA].fillna("PENDIENTE_PROCESAR")
    pendientes = df[df[COLUMNA_LIMPIA] == "PENDIENTE_PROCESAR"].index.tolist()

    if not pendientes:
        print("✅ No hay tareas pendientes.")
        return

    print(f"🚀 Procesando {len(pendientes)} comentarios. Logs en limpieza_corpus.log")
    pbar = tqdm(total=len(pendientes))

    ptr = 0
    while ptr < len(pendientes):
        lote_indices = []
        lote_objetos = []

        # Construcción del lote
        while len(lote_objetos) < 30 and ptr < len(pendientes):
            idx = pendientes[ptr]
            raw_text = str(df.at[idx, COLUMNA_ORIGINAL]).strip()

            # Limitar longitud para evitar desbordamiento de tokens (máx 2000 chars por comentario)
            raw_text = raw_text[:2000]

            if len(raw_text) < 2:
                df.at[idx, COLUMNA_LIMPIA] = "[CORTO_O_VACIO]"
            else:
                lote_objetos.append({"id": str(idx), "text": raw_text})

            ptr += 1
            pbar.update(1)

        if lote_objetos:
            # Llamada con Fallback inteligente
            mapa_res = procesar_lote_con_fallback(lote_objetos)

            for item in lote_objetos:
                uid = item['id']
                df.at[int(uid), COLUMNA_LIMPIA] = mapa_res.get(uid, "[ERROR_INESPERADO]")

            # Guardado Atómico (Safe Save)
            df.to_csv(OUTPUT_FILE + ".tmp", index=False, encoding='utf-8-sig')
            os.replace(OUTPUT_FILE + ".tmp", OUTPUT_FILE)

            # Respetar Rate Limit de 15 RPM (4.5s de delay es ideal)
            time.sleep(4.5)

    pbar.close()
    print(f"🏁 ¡Proceso Finalizado con éxito!")


if __name__ == "__main__":
    procesar_corpus()