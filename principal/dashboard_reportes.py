import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import re
import unicodedata
from sklearn.metrics import f1_score
try:
    from wordcloud import WordCloud, STOPWORDS
except ImportError:
    WordCloud = None
    STOPWORDS = set()


# ============================================================================
# FUNCIÓN UTILITARIA: ELIMINAR ACENTOS
# ============================================================================
def remove_accents(text: str) -> str:
    """
    Elimina acentos y diacríticos de un texto.
    ✅ CORRECCIÓN: Ahora PRESERVA la 'ñ' y 'Ñ'.
    """
    result = []
    for char in text:
        # Preservar explícitamente la eñe
        if char in ('ñ', 'Ñ'):
            result.append(char)
        else:
            nfkd = unicodedata.normalize('NFKD', char)
            stripped = ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))
            result.append(stripped)
    return ''.join(result)


# ============================================================================
# STOPWORDS EN ESPAÑOL - VERSIÓN ROBUSTA
# ============================================================================
try:
    import nltk

    nltk.data.find('corpora/stopwords')
    from nltk.corpus import stopwords as nltk_stopwords

    NLTK_SPANISH = set(nltk_stopwords.words('spanish'))
except:
    NLTK_SPANISH = set()

SPANISH_STOPWORDS_RAW = {
    # --- Artículos ---
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",

    # --- Preposiciones ---
    "a", "al", "del", "de", "en", "con", "por", "para", "sin",
    "sobre", "entre", "hacia", "hasta", "desde", "contra", "bajo",
    "durante", "mediante", "segun", "según", "tras", "via", "versus",

    # --- Conjunciones ---
    "y", "e", "o", "u", "ni", "pero", "aunque", "sino", "que",
    "porque", "como", "cuando", "mientras", "si", "sí",

    # --- Pronombres personales ---
    "yo", "tu", "tú", "el", "él", "ella", "ello", "nosotros",
    "nosotras", "vosotros", "vosotras", "ellos", "ellas", "me", "te",
    "le", "les", "nos", "os", "se",

    # --- Pronombres posesivos ---
    "mi", "mí", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",

    # --- Pronombres demostrativos ---
    "este", "esta", "esto", "estos", "estas", "ese", "esa", "eso",
    "esos", "esas", "aquel", "aquella", "aquello", "aquellos", "aquellas",
    "éste", "ésta", "éstos", "éstas", "ése", "ésa", "ésos", "ésas",
    "aquél", "aquélla", "aquéllos", "aquéllas",

    # --- Pronombres indefinidos ---
    "algo", "algun", "alguna", "algunas", "alguno", "algunos", "nadie",
    "ningun", "ninguna", "ninguno", "ningunos", "todo", "toda", "todos",
    "todas", "mucho", "mucha", "muchos", "muchas", "poco", "poca",
    "pocos", "pocas", "otro", "otra", "otros", "otras", "cualquier",
    "cualquiera", "demas", "demás", "varios", "varias", "sendos",

    # --- Pronombres relativos e interrogativos ---
    "que", "quien", "quién", "quiénes", "cual", "cuál", "cuáles",
    "cuando", "cuándo", "donde", "dónde", "como", "cómo", "cuanto",
    "cuánto", "cuánta", "cuántos", "cuántas",

    # --- Adverbios ---
    "no", "si", "sí", "ya", "muy", "bien", "mal", "aquí", "ahi", "ahí",
    "allí", "allá", "acá", "donde", "adonde", "bastante", "demasiado",
    "mas", "más", "menos", "tan", "tanto", "tal", "tales", "casi",
    "apenas", "quizas", "quizás", "acaso", "quizá", "talvez", "tal vez",
    "incluso", "ademas", "además", "también", "tambien", "nunca",
    "siempre", "jamás", "jamas", "ahora", "hoy", "ayer", "mañana",
    "antes", "despues", "después", "luego", "pronto", "entonces",
    "aun", "aún", "asi", "así", "pues", "entonces",

    # --- Verbos comunes (formas conjugadas) ---
    # Ser
    "ser", "soy", "eres", "es", "somos", "sois", "son", "era", "eras",
    "eramos", "éramos", "eran", "fui", "fuiste", "fue", "fuimos",
    "fuisteis", "fueron", "seré", "serás", "será", "seremos", "seréis",
    "serán", "sea", "seas", "seamos", "seáis", "sean", "fuera", "fueras",
    "fuéramos", "fueran", "fuese", "fueses", "fuésemos", "fuesen",
    "sido", "siendo",

    # Estar
    "estar", "estoy", "estás", "esta", "está", "estamos", "estáis",
    "están", "estaba", "estabas", "estabamos", "estábamos", "estaban",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis",
    "estuvieron", "estaré", "estarás", "estará", "estaremos", "estaréis",
    "estarán", "esté", "estés", "estemos", "estéis", "estén", "estuviera",
    "estuvieras", "estuviéramos", "estuvieran", "estado", "estando",

    # Tener
    "tener", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen",
    "tenía", "tenías", "teniamos", "teníamos", "tenían", "tuve", "tuviste",
    "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tendré", "tendrás",
    "tendrá", "tendremos", "tendréis", "tendrán", "tenga", "tengas",
    "tengamos", "tengáis", "tengan", "tuviera", "tuvieras", "tuviéramos",
    "tuvieran", "tenido", "teniendo",

    # Haber
    "haber", "he", "has", "ha", "hemos", "habéis", "han", "había",
    "habías", "habiamos", "habíamos", "habían", "hube", "hubiste", "hubo",
    "hubimos", "hubisteis", "hubieron", "habré", "habrás", "habrá",
    "habremos", "habréis", "habrán", "haya", "hayas", "hayamos", "hayáis",
    "hayan", "hubiera", "hubieras", "hubiéramos", "hubieran", "hay",
    "habido", "habiendo",

    # Hacer
    "hacer", "hago", "haces", "hace", "hacemos", "hacéis", "hacen",
    "hacía", "hacías", "haciamos", "hacíamos", "hacían", "hice", "hiciste",
    "hizo", "hicimos", "hicisteis", "hicieron", "haré", "harás", "hará",
    "haremos", "haréis", "harán", "haga", "hagas", "hagamos", "hagáis",
    "hagan", "hiciera", "hicieras", "hiciéramos", "hicieran", "hecho",
    "haciendo",

    # Poder
    "poder", "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "podía", "podías", "podiamos", "podíamos", "podían", "pude", "pudiste",
    "pudo", "pudimos", "pudisteis", "pudieron", "podré", "podrás", "podrá",
    "podremos", "podréis", "podrán", "pueda", "puedas", "podamos", "podáis",
    "puedan", "podido", "pudiendo",

    # Decir
    "decir", "digo", "dices", "dice", "decimos", "decís", "dicen",
    "decía", "decías", "deciamos", "decíamos", "decían", "dije", "dijiste",
    "dijo", "dijimos", "dijisteis", "dijeron", "diré", "dirás", "dirá",
    "diremos", "diréis", "dirán", "diga", "digas", "digamos", "digáis",
    "digan", "dicho", "diciendo",

    # Ir
    "ir", "voy", "vas", "va", "vamos", "vais", "van", "iba", "ibas",
    "ibamos", "íbamos", "iban", "fui", "fuiste", "fue", "fuimos",
    "fuisteis", "fueron", "iré", "irás", "irá", "iremos", "iréis", "irán",
    "vaya", "vayas", "vayamos", "vayáis", "vayan", "ido", "yendo",

    # Ver
    "ver", "veo", "ves", "ve", "vemos", "veis", "ven", "veía", "veías",
    "veiamos", "veíamos", "veían", "vi", "viste", "vio", "vimos",
    "visteis", "vieron", "veré", "verás", "verá", "veremos", "veréis",
    "verán", "visto", "viendo",

    # Dar
    "dar", "doy", "das", "da", "damos", "dais", "dan", "daba", "dabas",
    "dabamos", "dábamos", "daban", "di", "diste", "dio", "dimos",
    "disteis", "dieron", "dado", "dando",

    # Saber
    "saber", "sé", "sabes", "sabe", "sabemos", "sabéis", "saben",
    "sabía", "sabías", "sabiamos", "sabíamos", "sabían", "supe", "supiste",
    "supo", "supimos", "supisteis", "supieron", "sabido", "sabiendo",

    # Querer
    "querer", "quiero", "quieres", "quiere", "queremos", "queréis",
    "quieren", "quería", "querías", "queriamos", "queríamos", "querían",
    "quise", "quisiste", "quiso", "quisimos", "quisisteis", "quisieron",
    "querido", "queriendo",

    # Llegar
    "llegar", "llego", "llegas", "llega", "llegamos", "llegáis", "llegan",
    "llegaba", "llegabas", "llegabamos", "llegaban", "llegué", "llegaste",
    "llegó", "llegamos", "llegasteis", "llegaron", "llegado", "llegando",

    # Otros verbos comunes
    "parecer", "parece", "parecio", "pareció", "parecía", "parecían",
    "quedar", "queda", "quedan", "quedaba", "quedaban",
    "poner", "pongo", "pone", "ponen", "ponía", "ponían", "puesto",
    "seguir", "sigo", "sigue", "siguen", "seguía", "seguían", "seguido",
    "encontrar", "encuentro", "encuentra", "encuentran", "encontrado",
    "creer", "creo", "cree", "creen", "creía", "creían", "creído",
    "deber", "debo", "debe", "deben", "debía", "debían", "debido",
    "dejar", "dejo", "deja", "dejan", "dejaba", "dejaban", "dejado",
    "llamar", "llamo", "llama", "llaman", "llamaba", "llamaban", "llamado",
    "venir", "vengo", "viene", "vienen", "venía", "venían", "venido",
    "pasar", "paso", "pasa", "pasan", "pasaba", "pasaban", "pasado",
    "llevar", "llevo", "lleva", "llevan", "llevaba", "llevaban", "llevado",
    "traer", "traigo", "trae", "traen", "traía", "traían", "traído",
    "salir", "salgo", "sale", "salen", "salía", "salían", "salido",
    "funcionar", "funciona", "funcionan", "funcionaba", "funcionado",
    "servir", "sirvo", "sirve", "sirven", "servía", "servían", "servido",

    # --- Otras palabras vacías ---
    "cada", "cabe", "solo", "solamente", "propio", "propios", "propia",
    "propias", "mismo", "mismos", "misma", "mismas", "vez", "veces",
    "gente", "forma", "manera", "parte", "lugar", "tiempo", "momento",
    "punto", "caso", "ejemplo", "tipo", "clase", "todavia", "todavía",
    "excepto", "general", "particular", "especial", "normal", "cierto",
    "posible", "imposible", "seguro", "segura",
}


# ============================================================================
# CONSTRUIR SET FINAL DE STOPWORDS (CON NORMALIZACIÓN DE ACENTOS)
# ============================================================================
def build_stopwords_set():
    """
    Construye el set final de stopwords, incluyendo versiones con y sin acentos.
    """
    final_set = set()

    # 1. Stopwords de WordCloud (inglés)
    final_set.update(STOPWORDS)

    # 2. Stopwords de NLTK si están disponibles
    final_set.update(NLTK_SPANISH)

    # 3. Lista manual
    final_set.update(SPANISH_STOPWORDS_RAW)

    # 4. NORMALIZAR: Para cada palabra, agregar su versión sin acentos
    words_to_add = set()
    for word in list(final_set):
        normalized = remove_accents(word.lower())
        words_to_add.add(normalized)

    final_set.update(words_to_add)

    # 5. Todo en minúsculas
    final_set = {w.lower() for w in final_set}

    return final_set


SPANISH_STOPWORDS = build_stopwords_set()

# ============================================================================
# STOPWORDS CONTEXTUALES (solo RUIDO REAL, sin quitar términos de negocio)
# ============================================================================
CONTEXT_STOPWORDS_RAW = {
    # --- Plataforma (no aportan valor analítico) ---
    "mercado", "libre", "ml",

    # --- Transacción/Logística (ruido de proceso, no del producto) ---
    "compra", "comprado", "comprar", "compre", "pedido", "entrega",
    "llegada", "llego", "llegó", "recibido", "recibi", "recibí",
    "envio", "envío", "paquete", "caja", "empaque", "vendedor",
    "tienda", "enviar", "enviaron",

    # --- Verbos de opinión (no describen el producto) ---
    "parece", "parecio", "pareció", "creo", "opino", "opinion",
    "opinión", "parecer", "pensar", "pienso", "considero", "parezco",

    # --- Recomendaciones genéricas (no describen el producto) ---
    "recomiendo", "recomiendo", "recomendable", "recomendado",
    "recomendar", "sugerir", "sugiero", "sugerencia",

    # --- Adjetivos de valoración SIN descripción específica ---
    # (Son tan genéricos que no ayudan a identificar el problema)
    "bonito", "bonita", "bueno", "buena", "buen", "excelente",
    "perfecto", "perfecta", "genial", "increible", "increíble",
    "maravilloso", "maravillosa", "fantastico", "fantástico",
    "feo", "fea", "terrible", "horrible", "pesimo", "pésimo", "fatal",

    # --- Palabras de relleno ---
    "verdad", "realidad", "hecho", "nada", "algo", "bastante",
    "suficiente", "necesario", "varios", "varias",

    # =========================================================================
    # ⚠️  PALABRAS COMENTADAS - Decidir caso por caso según el objetivo:
    # =========================================================================

    # --- Palabras de producto (podrían ser útiles para frecuencia) ---
    # "laptop", "computadora", "compu", "equipo", "producto", "articulo",
    # "artículo", "maquina", "máquina", "dispositivo", "aparato",

    # --- Palabras de precio/calidad (ÚTILES en quejas) ---
    # "precio", "costo", "dinero", "pagar", "pago", "barato", "caro",
    # "calidad", "relacion", "relación",

    # --- Palabras de PROBLEMA (CLAVE en quejas - NO eliminar) ---
    # "problema", "defecto", "detalle", "falla", "error", "fallo",
}

# Normalizar contextuales (sin acentos)
CONTEXT_STOPWORDS = {remove_accents(w.lower()) for w in CONTEXT_STOPWORDS_RAW}


# ============================================================================
# FUNCIÓN PARA PREPARAR TEXTO PARA LA NUBE DE PALABRAS
# ============================================================================
def prepare_text_for_wordcloud(text_series: pd.Series) -> str:
    """
    Prepara una serie de textos para la nube de palabras:
    1. Convierte a minúsculas
    2. Elimina acentos (para que coincidan con stopwords normalizadas)
    3. Elimina caracteres especiales y números
    4. Une todo en un solo string
    """

    def clean_single_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = remove_accents(text)  # 'ñ' → 'n', 'á' → 'a', etc.
        # Eliminar todo excepto letras (a-z) y espacios
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    cleaned_texts = text_series.apply(clean_single_text)
    return " ".join(cleaned_texts)


# ============================================================================
# FUNCIÓN AUXILIAR: CARGAR IMAGEN DEL SIDEBAR (con fallback)
# ============================================================================
def load_sidebar_image():
    """
    Intenta cargar el logo desde archivo local primero;
    si no existe, usa la URL remota.
    """
    logo_local = get_path("logo_fit.png")
    logo_url = "https://www.uat.edu.mx/itampico/SiteAssets/Logo_FIT.png"

    if os.path.exists(logo_local):
        return logo_local
    return logo_url


# ============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Dashboard Pro: Inteligencia Emocional",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


# ============================================================================
# 2. CARGA DE DATOS (con validación de existencia y encoding robusto)
# ============================================================================
@st.cache_data
def load_data():
    # CORRECCIÓN: Apuntar al archivo final que genera todo el pipeline
    path = get_path("corpus_limpio_final_fase3_CORREGIDO_fase4_completo.csv")

    if not os.path.exists(path):
        st.error(
            f"❌ **Archivo no encontrado:** `{path}`\n\n"
            "Asegúrate de haber ejecutado los scripts fases 4a y 4b antes de abrir el Dashboard."
        )
        st.stop()

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


@st.cache_data
def load_predictions():
    path = get_path("predicciones_test.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def sanitize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "") # Limpia el BOM si existe
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()


# CARGAR DATOS
df = load_data()
df['final_clean_text'] = df['final_clean_text'].apply(sanitize_text)

# Normalizar intensidad
df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
df = df[df['intensity'].between(1, 4)]

# Convertir intensidad a texto
df['Intensidad_Cat'] = df['intensity'].astype(str)

# ============================================================================
# 3. MAPA DE COLORES DE ALTO CONTRASTE
# ============================================================================
# ✅ CORRECCIÓN #1: Agregar 'Neutral' al mapa de colores
color_emociones = {
    'Alegría': '#2ca02c',
    'Neutral': '#17becf',       # ✅ AGREGADO — era la emoción faltante
    'Tristeza': '#1f77b4',
    'Enojo': '#d62728',
    'Sorpresa': '#ff7f0e',
    'Miedo': '#9467bd',
    'Asco': '#8c564b'
}

color_intensidad = {
    '1': '#00cc96',
    '2': '#fecb52',
    '3': '#EF553B',
    '4': '#b22222'
}

# ============================================================================
# 4. SIDEBAR (PANEL LATERAL FIJO)
# ============================================================================
# ✅ CORRECCIÓN #5: Logo con fallback local → remoto
st.sidebar.image(load_sidebar_image(), width=200)

st.sidebar.title("🎮 Panel de Control")
st.sidebar.markdown("Usa estos filtros para segmentar el corpus.")
st.sidebar.divider()

# Filtros dinámicos en el Sidebar
marcas_disp = sorted(df['marca'].dropna().unique())
emociones_disp = sorted(df['emotion'].dropna().unique())

f_marcas = st.sidebar.multiselect("🏷️ Seleccionar Marcas:", options=marcas_disp)
modelos_base = df[df['marca'].isin(f_marcas)] if f_marcas else df
modelos_disp = sorted(modelos_base['modelo'].dropna().unique())
f_modelos = st.sidebar.multiselect("🧩 Seleccionar Modelos:", options=modelos_disp)
f_emociones = st.sidebar.multiselect("🎭 Seleccionar Emociones:", options=emociones_disp)

# ✅ CORRECCIÓN #2: Proteger contra select_slider devolviendo int en lugar de tupla
f_intensidad = st.sidebar.select_slider(
    "🔥 Nivel de Intensidad:",
    options=[1, 2, 3, 4],
    value=(1, 4)
)
if isinstance(f_intensidad, int):
    f_intensidad = (f_intensidad, f_intensidad)

f_keyword = st.sidebar.text_input("🔎 Buscar palabra clave:").strip()

# Aplicar filtros
df_view = df.copy()
if f_marcas:
    df_view = df_view[df_view['marca'].isin(f_marcas)]
if f_modelos:
    df_view = df_view[df_view['modelo'].isin(f_modelos)]
if f_emociones:
    df_view = df_view[df_view['emotion'].isin(f_emociones)]
if f_keyword:
    df_view = df_view[
        df_view['final_clean_text'].str.contains(f_keyword, case=False, na=False)
    ]
df_view = df_view[
    (df_view['intensity'] >= f_intensidad[0]) &
    (df_view['intensity'] <= f_intensidad[1])
]
df_view['Intensidad_Cat'] = df_view['intensity'].astype(str)

# ============================================================================
# 5. CUERPO PRINCIPAL
# ============================================================================
st.title("📊 Dashboard de Sentimientos: Laptops Mercado Libre")
st.markdown(f"Visualización de **{len(df_view):,}** registros filtrados.")

# ============================================================================
# 6. MÉTRICAS RÁPIDAS (calculadas dinámicamente)
# ============================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros Actuales", f"{len(df_view):,}")

# ✅ CORRECCIÓN #6: Calcular Exactitud SVM desde las predicciones reales
pred_df = load_predictions()
if pred_df is not None and len(pred_df) > 0:
    acc_real = (pred_df['emotion_real'] == pred_df['emotion_pred']).mean()
    m2.metric("Exactitud SVM", f"{acc_real:.1%}")
else:
    m2.metric("Exactitud SVM", "—")

# ✅ CORRECCIÓN #6b: Calcular F1-Score promedio desde las predicciones reales
if pred_df is not None and len(pred_df) > 0:
    try:
        f1_val = f1_score(
            pred_df['emotion_real'],
            pred_df['emotion_pred'],
            average='weighted',
            zero_division=0
        )
        m3.metric("F1-Score (Ponderado)", f"{f1_val:.2f}")
    except Exception:
        m3.metric("F1-Score (Ponderado)", "—")
else:
    m3.metric("F1-Score (Ponderado)", "—")

m4.metric("Alertas Críticas (Nivel 4)", len(df_view[df_view['intensity'] == 4]))

# ============================================================================
# 7. TABS PRINCIPALES
# ============================================================================
tab_analisis, tab_negocio, tab_eval, tab_datos = st.tabs([
    "📈 Análisis Estadístico",
    "💼 Casos de Negocio",
    "🧪 Evaluación del Modelo",
    "📋 Explorador"
])

# ============================================================================
# TAB: ANÁLISIS ESTADÍSTICO
# ============================================================================
with tab_analisis:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Frecuencia por Emoción")
        total_view = max(len(df_view), 1)
        emotion_counts = df_view['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        emotion_counts['percent'] = (emotion_counts['count'] / total_view * 100).round(2)
        top_brand = (
            df_view.groupby(['emotion', 'marca'])
            .size()
            .reset_index(name='count')
            .sort_values(['emotion', 'count'], ascending=[True, False])
            .drop_duplicates('emotion')
        )
        emotion_counts = emotion_counts.merge(
            top_brand[['emotion', 'marca']],
            on='emotion',
            how='left'
        ).rename(columns={'marca': 'top_brand'})

        fig1 = px.bar(
            emotion_counts,
            x="emotion",
            y="count",
            text="count",
            color="emotion",
            color_discrete_map=color_emociones,  # ✅ Ahora incluye 'Neutral'
            hover_data={"percent": ":.2f", "top_brand": True, "count": False}
        )
        fig1.update_traces(textposition='outside', cliponaxis=False)
        fig1.update_layout(showlegend=False, xaxis_title="Emoción", yaxis_title="Cantidad")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Distribución de Intensidad")
        fig2 = px.pie(
            df_view,
            names="Intensidad_Cat",
            hole=0.5,
            color_discrete_map=color_intensidad
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Análisis Cruzado: Emoción vs Intensidad")
    df_group = df_view.groupby(['emotion', 'Intensidad_Cat']).size().reset_index(name='Reseñas')
    fig3 = px.bar(
        df_group,
        x="emotion",
        y="Reseñas",
        color="Intensidad_Cat",
        barmode="group",
        text_auto=True,
        category_orders={"Intensidad_Cat": ["1", "2", "3", "4"]},
        color_discrete_map=color_intensidad
    )
    fig3.update_traces(textposition='outside', cliponaxis=False)
    fig3.update_layout(
        xaxis_title="Emoción",
        yaxis_title="Cantidad de Reseñas",
        legend_title="Intensidad"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# TAB: CASOS DE NEGOCIO
# ============================================================================
with tab_negocio:
    st.header("🎯 Aplicaciones Estratégicas")
    c_qa, c_wc, c_sla = st.columns(3)

    with c_qa:
        st.subheader("Top 5 Modelos con más alertas")
        df_qa = df_view[
            (df_view['emotion'].isin(['Enojo', 'Miedo'])) &
            (df_view['intensity'] >= 3)
        ]
        top_qa = (
            df_qa.groupby('modelo')
            .size()
            .reset_index(name='alertas')
            .sort_values('alertas', ascending=False)
            .head(5)
        )
        if not top_qa.empty:
            fig_qa = px.bar(
                top_qa,
                x='alertas',
                y='modelo',
                orientation='h',
                text='alertas'
            )
            fig_qa.update_traces(textposition='outside', cliponaxis=False)
            fig_qa.update_layout(
                xaxis_title='Alertas (Intensidad 3-4)',
                yaxis_title='Modelo'
            )
            st.plotly_chart(fig_qa, use_container_width=True)
        else:
            st.info("Sin alertas en el filtro actual.")

    # ========================================================================
    # NUBE DE PALABRAS
    # ========================================================================
    with c_wc:
        st.subheader("Nube de palabras (Quejas)")
        df_wc = df_view[df_view['emotion'].isin(['Enojo', 'Tristeza'])]

        if WordCloud and not df_wc.empty:
            text_blob = prepare_text_for_wordcloud(df_wc['final_clean_text'])
            all_stopwords = STOPWORDS | SPANISH_STOPWORDS | CONTEXT_STOPWORDS

            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color='white',
                stopwords=all_stopwords,
                collocations=False,
                min_word_length=3,
                max_words=100,
                prefer_horizontal=0.7,
                regexp=r'[a-z]{3,}'
            ).generate(text_blob)

            fig_wc, ax_wc = plt.subplots(figsize=(7, 4.5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout()
            st.pyplot(fig_wc, use_container_width=True)

        elif df_wc.empty:
            st.info("No hay textos de queja en el filtro actual.")
        else:
            st.warning("⚠️ Instala 'wordcloud' para visualizar la nube de palabras.")
            st.code("pip install wordcloud", language="bash")

    with c_sla:
        st.subheader("SLA Crítico (Intensidad 4)")
        total_rows = max(len(df_view), 1)
        crit_count = int((df_view['intensity'] == 4).sum())
        crit_pct = round((crit_count / total_rows) * 100, 2)
        fig_sla = go.Figure(go.Indicator(
            mode="gauge+number",
            value=crit_pct,
            number={'suffix': '%'},
            gauge={'axis': {'range': [0, 100]}}
        ))
        fig_sla.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sla, use_container_width=True)

    with st.expander("Ver reportes detallados"):
        st.write("### QA Hardware")
        st.dataframe(
            df_qa[['marca', 'modelo', 'final_clean_text', 'intensity']],
            use_container_width=True
        )

        st.write("### Soporte SLA")
        df_sup = df_view[df_view['intensity'] == 4]
        st.dataframe(
            df_sup[['marca', 'final_clean_text', 'intensity']],
            use_container_width=True
        )

        st.write("### Marketing (Alegría Extrema)")
        df_mkt = df_view[
            (df_view['emotion'] == 'Alegría') & (df_view['intensity'] == 4)
        ]
        st.dataframe(
            df_mkt[['marca', 'modelo', 'final_clean_text']],
            use_container_width=True
        )

# ============================================================================
# TAB: EVALUACIÓN DEL MODELO
# ============================================================================
with tab_eval:
    st.header("🧪 Evaluación del Modelo y Estadística")

    col_a, col_b = st.columns(2)

    # ✅ CORRECCIÓN #7: Calcular anomalías de sarcasmo desde el CSV real
    if 'possible_sarcasm' in df_view.columns:
        sarcasmo_count = int(df_view['possible_sarcasm'].sum())
    else:
        sarcasmo_count = 0
    col_a.metric("Anomalías Semánticas (Sarcasmo) Detectadas", sarcasmo_count)
    col_b.metric("Registros Evaluados", f"{len(df_view):,}")

    st.subheader("Matriz de Confusión (SVM)")
    if pred_df is None:
        st.warning(
            "No se encontró `predicciones_test.csv`. "
            "Ejecuta `fase_5.py` para generarlo."
        )
    else:
        cm = pd.crosstab(pred_df['emotion_real'], pred_df['emotion_pred'])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(
            xaxis_title='Predicción',
            yaxis_title='Etiqueta Real'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Estadísticas Descriptivas por Marca")
    stats = (
        df_view.groupby('marca')['intensity']
        .agg(promedio='mean', varianza='var', desviacion='std', conteo='count')
        .reset_index()
        .sort_values('promedio', ascending=False)
    )
    st.dataframe(stats, use_container_width=True)

# ============================================================================
# TAB: EXPLORADOR DE DATOS
# ============================================================================
with tab_datos:
    st.subheader("Explorador de Reseñas Sanitizadas")
    st.dataframe(
        df_view[[
            'id_review', 'marca', 'modelo',
            'final_clean_text', 'emotion', 'intensity'
        ]],
        use_container_width=True,
        height=600
    )

# ============================================================================
# FOOTER DEL SIDEBAR
# ============================================================================
st.sidebar.divider()
st.sidebar.caption("Proyecto Final | Equipo 3 | UAT 2026")