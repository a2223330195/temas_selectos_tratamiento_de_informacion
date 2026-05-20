import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # <-- CORRECCIÓN: Agregar esta línea
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS A PRUEBA DE FALLOS ---
BASE_DIR = Path(__file__).resolve().parent
archivo_csv = BASE_DIR / 'corpus_limpio_final_fase3_CORREGIDO_fase4_completo.csv'

# 1. Cargar el dataset de la Fase 3
try:
    # Usamos utf-8-sig como primer intento por los caracteres especiales
    df = pd.read_csv(archivo_csv, encoding='utf-8-sig')
except UnicodeDecodeError:
    # Fallback clásico
    df = pd.read_csv(archivo_csv, encoding='latin1')

# Asegurarnos de que no hay valores nulos o vacíos en texto y etiquetas
df['final_clean_text'] = df['final_clean_text'].fillna('').astype(str).str.strip()
df['emotion'] = df['emotion'].fillna('').astype(str).str.strip()
df = df[df['final_clean_text'].str.len() > 0]
df = df[df['emotion'].str.len() > 0]

# Eliminar clases con menos de 2 ejemplos antes de estratificar
valid_emotions = df['emotion'].value_counts()
valid_emotions = valid_emotions[valid_emotions >= 2].index
df = df[df['emotion'].isin(valid_emotions)]

X = df['final_clean_text']
y = df['emotion']

# 2. Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")

# 3. Vectorización del texto (TF-IDF)
# Convierte las palabras en números que la máquina pueda entender
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Entrenar el modelo SVM
# class_weight='balanced' es crucial aquí por el desbalanceo extremo hacia 'Alegría'
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
print("\nEntrenando el modelo SVM... (Esto puede tomar unos segundos)")
svm_model.fit(X_train_vec, y_train)
print("Entrenamiento finalizado.")

# 5. Evaluar el modelo
y_pred = svm_model.predict(X_test_vec)

# 5.1 Exportar predicciones del conjunto de prueba
export_cols = ['final_clean_text', 'emotion']
if 'id_review' in df.columns:
    export_cols.insert(0, 'id_review')

df_test = df.loc[X_test.index, export_cols].copy()
df_test = df_test.rename(columns={'emotion': 'emotion_real'})
df_test['emotion_pred'] = y_pred
pred_path = BASE_DIR / 'predicciones_test.csv'
df_test.to_csv(pred_path, index=False, encoding='utf-8-sig')
print(f"\nPredicciones del test guardadas en: {pred_path}")

print("\n--- REPORTE DE CLASIFICACIÓN (FASE 6) ---")
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

# 6. Generar Matriz de Confusión
cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.title('Matriz de Confusión - Modelo Base SVM', fontsize=14, fontweight='bold')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Etiqueta Real')
plt.tight_layout()

# Guardar la gráfica en la misma ruta del script
ruta_grafica = BASE_DIR / 'grafica_5_matriz_confusion.png'
plt.savefig(ruta_grafica, dpi=300)
print(f"\nMatriz de confusión guardada exitosamente en: {ruta_grafica}")