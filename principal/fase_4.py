import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Configurar el estilo de las gráficas
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 2. Cargar el dataset etiquetado (Fase 3)
BASE_DIR = Path(__file__).resolve().parent
archivo_csv = BASE_DIR / 'corpus_limpio_final_fase3_CORREGIDO_fase4_completo.csv'  # Cambia el nombre si tu archivo se llama distinto

df = None
for enc in ('utf-8-sig', 'utf-8', 'latin1'):
    try:
        df = pd.read_csv(archivo_csv, encoding=enc)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise RuntimeError(f"No se pudo leer el archivo {archivo_csv} con las codificaciones probadas.")

print(f"Dataset cargado exitosamente. Total de reseñas: {df.shape[0]}")

# Asegurar que la intensidad es numérica y válida (1-4)
df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
df = df[df['intensity'].between(1, 4)]

# ==========================================
# GRÁFICA 1: Distribución General de Emociones
# ==========================================
plt.figure(figsize=(10, 6))
# CORRECCIÓN: Se eliminó hue='emotion' y legend=False
ax1 = sns.countplot(data=df, x='emotion', hue='emotion', order=df['emotion'].value_counts().index, palette='viridis', legend=False)
plt.title('Distribución General de Emociones en las Reseñas', fontsize=14, fontweight='bold')
plt.xlabel('Emoción', fontsize=12)
plt.ylabel('Cantidad de Reseñas', fontsize=12)

# Agregar los números arriba de cada barra
for p in ax1.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        ax1.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_1_emociones.png', dpi=300)
plt.show()

# ==========================================
# GRÁFICA 2: Distribución de Niveles de Intensidad
# ==========================================
plt.figure(figsize=(8, 5))
# CORRECCIÓN: Se eliminó hue='intensity' y legend=False
ax2 = sns.countplot(data=df, x='intensity', hue='intensity', order=[1, 2, 3, 4], palette='magma', legend=False)
plt.title('Distribución de Niveles de Intensidad (1 al 4)', fontsize=14, fontweight='bold')
plt.xlabel('Nivel de Intensidad', fontsize=12)
plt.ylabel('Cantidad de Reseñas', fontsize=12)

for p in ax2.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        ax2.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_2_intensidad.png', dpi=300)
plt.show()

# ==========================================
# GRÁFICA 3: Emociones desglosadas por Intensidad
# ==========================================
plt.figure(figsize=(12, 6))
ax3 = sns.countplot(data=df, x='emotion', hue='intensity', order=df['emotion'].value_counts().index, palette='coolwarm')
plt.title('Emociones vs. Nivel de Intensidad', fontsize=14, fontweight='bold')
plt.xlabel('Emoción', fontsize=12)
plt.ylabel('Cantidad de Reseñas', fontsize=12)
plt.legend(title='Intensidad')

# NUEVO: Agregar números a las barras agrupadas
for p in ax3.patches:
    height = p.get_height()
    # Solo mostramos el número si la barra tiene un valor mayor a 0
    if pd.notnull(height) and height > 0:
        ax3.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_3_emocion_vs_intensidad.png', dpi=300)
plt.show()

# ==========================================
# GRÁFICA 4: Emociones por Marca de Laptop
# ==========================================
if 'marca' in df.columns:
    plt.figure(figsize=(14, 7))
    ax4 = sns.countplot(data=df, x='marca', hue='emotion', palette='Set2')
    plt.title('Distribución de Emociones por Marca de Laptop', fontsize=14, fontweight='bold')
    plt.xlabel('Marca', fontsize=12)
    plt.ylabel('Cantidad de Reseñas', fontsize=12)
    plt.legend(title='Emoción', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)

    # NUEVO: Agregar números a las barras por marca
    for p in ax4.patches:
        height = p.get_height()
        # Solo mostramos el número si la barra tiene un valor mayor a 0
        if pd.notnull(height) and height > 0:
            ax4.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'grafica_4_emocion_por_marca.png', dpi=300)
    plt.show()

# ==========================================
# ESTADÍSTICAS BÁSICAS (Para su Reporte Documental)
# ==========================================
print("\n--- RESUMEN ESTADÍSTICO PARA EL REPORTE ---")
print("\nPorcentajes por Emoción:")
print(round(df['emotion'].value_counts(normalize=True) * 100, 2).astype(str) + '%')

print("\nPorcentajes por Intensidad:")
print(round(df['intensity'].value_counts(normalize=True) * 100, 2).astype(str) + '%')