from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time, random

import re

# ─── Configura el driver ───────────────────────────────────────────────
options = webdriver.ChromeOptions()
options.add_argument('--lang=es-MX')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
# Quita el comentario de abajo si quieres que corra sin abrir ventana
# options.add_argument('--headless')

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# ─── PEGA AQUÍ TUS URLs REALES DE MERCADO LIBRE ───────────────────────
productos = [
    {"url": "https://www.mercadolibre.com.mx/laptop-hp-14-dq0518la-celeron-n4120-ram-4gb-ssd-128gb-w11h-color-plateado/p/MLM27712091", "id": "ML_PROD_001", "marca": "HP",     "modelo": "HP 14"},
    {"url": "https://www.mercadolibre.com.mx/hp-pink-stream-14-ep2011-intel-n150-4-gb-ram-128-gb-ufs-incluye-windows-11-instalado-pantalla-14-laptop-rosa-con-regalo/p/MLM53196537", "id": "ML_PROD_002", "marca": "HP",     "modelo": "HP Stream 14"},
    {"url": "https://www.mercadolibre.com.mx/laptop-lenovo-ideapad-celeron-4gb-128ssd-office-regalo-color-gris/p/MLM27986873", "id": "ML_PROD_003", "marca": "Lenovo", "modelo": "Lenovo Slim 3"},
    {"url": "https://www.mercadolibre.com.mx/laptop-lenovo-ip-slim-3-ryzen-3-7320u-8-gb-512-ssd-abyss-blue/p/MLM47605348", "id": "ML_PROD_004", "marca": "Lenovo", "modelo": "Lenovo IdeaPad"},
    {"url": "https://www.mercadolibre.com.mx/laptop-dell-inspiron-3535-amd-ryzen-5-7520u-8gb-ram-512gb-ssd-156-windows-11-home-teclado-en-espanol/p/MLM37175361", "id": "ML_PROD_005", "marca": "Dell",   "modelo": "Dell Inspiron"},
    {"url": "https://www.mercadolibre.com.mx/laptop-acer-aspire-3-amd-ryzen-7-16gb-ram-512gb-ssd-156-windows-11-home/p/MLM34680070", "id": "ML_PROD_006", "marca": "Acer",   "modelo": "Acer Aspire 3"},
    {"url": "https://www.mercadolibre.com.mx/laptop-asus-vivobook-14-intel-core-i3-8gb-ram-128gb-ssd-windows-11-home/p/MLM50871366", "id": "ML_PROD_007", "marca": "ASUS",   "modelo": "ASUS Vivobook 14"},
    {"url": "https://www.mercadolibre.com.mx/laptop-hp-stream-14-dq0762dx-rosa-128gb-ssd-intel-celeron-incluye-windows-color-pink/p/MLM37919936", "id": "ML_PROD_008", "marca": "HP",     "modelo": "HP Stream 14 Rosa"},
    {"url": "https://www.mercadolibre.com.mx/laptop-gamer-hp-victus-156-amd-ryzen-7-7445hs-16gb-ram-512-ssd-nvidia-geforce-rtx-4050/p/MLM53049202", "id": "ML_PROD_009", "marca": "HP",     "modelo": "HP Victus 15"},
    {"url": "https://www.mercadolibre.com.mx/notebook-hp-14-dq6015dx-14-hd-128gb-4gb-ram-intel-n150-w11-pale-rose-gold/p/MLM58343227", "id": "ML_PROD_010", "marca": "HP",     "modelo": "HP 14-dq6015dx"},
    {"url": "https://www.mercadolibre.com.mx/notebook-elitebook-845-g8-14-plateado-256gb-ssd-amd-ryzen-5-excelente-reacondicionado/p/MLM2020089212", "id": "ML_PROD_011", "marca": "HP",     "modelo": "HP EliteBook 845 G8"},
    {"url": "https://www.mercadolibre.com.mx/notebook-asus-tuf-a16-16-fhd-512gb-16gb-ram-ryzen-7-7445hs-rtx4050-mecha-gray/p/MLM61511567", "id": "ML_PROD_012", "marca": "ASUS",   "modelo": "ASUS TUF A16"},
    {"url": "https://www.mercadolibre.com.mx/notebook-coolby-156-intel-n95-12gb-ram-1tb-ssd-full-hd-windows-11-pro/p/MLM44443027", "id": "ML_PROD_013", "marca": "Coolby", "modelo": "Coolby 15.6"},
    {"url": "https://www.mercadolibre.com.mx/laptop-gamer-loq-intel-core-i5-24gb-512gb-ssd-rtx-5050-pc-ia-gris-luna-lenovo/p/MLM53289250", "id": "ML_PROD_014", "marca": "Lenovo", "modelo": "Lenovo LOQ Gamer"},
    {"url": "https://www.mercadolibre.com.mx/laptop-dell-15-dc15255-amd-ryzen-5-8gb-512gb-156-fhd-windows-11-home-platinum-silver/p/MLM55283791", "id": "ML_PROD_015", "marca": "Dell",   "modelo": "Dell 15"},
    {"url": "https://www.mercadolibre.com.mx/apple-macbook-air-136-ips-chip-m4-256gb-ssd-16gb-ram-color-gris-oscuro/p/MLM53449012", "id": "ML_PROD_016", "marca": "Apple",  "modelo": "MacBook Air M4"},
    {"url": "https://www.mercadolibre.com.mx/macbook-air-a1466-plata-133-intel-core-i5-5350u-8gb-de-ram-128gb-ssd-hd-graphics-6000-1440x900px-macos-excelente-reacondicionado/p/MLM2010682657", "id": "ML_PROD_017", "marca": "Apple",  "modelo": "MacBook Air A1466"},
    {"url": "https://www.mercadolibre.com.mx/apple-macbook-air-m2-2022-mly03lla-midnight-136-2560-px-x-1664-px-apple-m2-8gb-de-ram/p/MLM19563442", "id": "ML_PROD_018", "marca": "Apple",  "modelo": "MacBook Air M2"},
    {"url": "https://www.mercadolibre.com.mx/laptop-huawei-matebook-d-16-i5-12th-8gb-512gb-win11-gris/p/MLM36412809", "id": "ML_PROD_019", "marca": "Huawei", "modelo": "Huawei MateBook D 16"},
    {"url": "https://www.mercadolibre.com.mx/laptop-gamer-msi-thin-15-nvidia-geforce-rtx-4050-intel-core-i5-16gb-ram-512gb-ssd-windows-11-home/p/MLM54057643", "id": "ML_PROD_020", "marca": "MSI",    "modelo": "MSI Thin 15"},
    {"url": "https://www.mercadolibre.com.mx/laptop-asus-vivobook-go-15-r5-16gb-512gb-mochila-mouse/p/MLM34348752", "id": "ML_PROD_021", "marca": "ASUS",   "modelo": "ASUS Vivobook Go 15"},
    {"url": "https://www.mercadolibre.com.mx/laptop-mochila-vivobook-ci31315u-8gb-ram-512ssd-asus/p/MLM62814295", "id": "ML_PROD_022", "marca": "ASUS",   "modelo": "ASUS Vivobook CI3"},
    {"url": "https://www.mercadolibre.com.mx/laptop-asus-vivobook-go-intel-celeron-n4500-4gb-ram-128-gb-ssd/p/MLM52115064", "id": "ML_PROD_023", "marca": "ASUS",   "modelo": "ASUS Vivobook Go Celeron"},
    {"url": "https://www.mercadolibre.com.mx/laptop-asus-vivobook-14-intel-core-i7-12gb-ram-512gb-ssd-windows-11-home/p/MLM49901787", "id": "ML_PROD_024", "marca": "ASUS",   "modelo": "ASUS Vivobook 14 i7"},
    {"url": "https://www.mercadolibre.com.mx/laptop-vivobook-16-amd-r7-16gb-512gb-ssd-asus/p/MLM34115956", "id": "ML_PROD_025", "marca": "ASUS",   "modelo": "ASUS Vivobook 16"},
    {"url": "https://www.mercadolibre.com.mx/laptop-lenovo-ip-slim-5-ia-ultra-7-16gb-512ssd-fhd/p/MLM44594668", "id": "ML_PROD_026", "marca": "Lenovo", "modelo": "Lenovo IdeaPad Slim 5"},
    {"url": "https://www.mercadolibre.com.mx/laptop-gamer-lenovo-loq-nvidia-geforce-rtx-3050-amd-ryzen-5-8gb-ram-512gb-ssd-windows-11-home/p/MLM41514241", "id": "ML_PROD_027", "marca": "Lenovo", "modelo": "Lenovo LOQ RTX 3050"},
    {"url": "https://www.mercadolibre.com.mx/laptop-dell-inspiron-15-3530-intel-core-i5-1334u-512gb-ssd-16gb-ram-1920x1080px-windows-11-home/p/MLM59102591", "id": "ML_PROD_028", "marca": "Dell",   "modelo": "Dell Inspiron 15 3530"},
    {"url": "https://www.mercadolibre.com.mx/laptop-inspiron-3530-ci5-1334u-16gb-ram-512gb-ssd-156-dell/p/MLM44555204", "id": "ML_PROD_029", "marca": "Dell",   "modelo": "Dell Inspiron 3530 CI5"},
]

MAX_RESENAS = 400  # máximo de reseñas por producto (meta: ~5000 total)


def obtener_url_resenas(soup):
    """Busca en la página del producto el enlace directo a todas las reseñas."""
    # Buscar enlace con href que contenga /noindex/catalog/reviews/
    link = soup.find('a', href=re.compile(r'/noindex/catalog/reviews/'))
    if link:
        return 'https://www.mercadolibre.com.mx' + link['href']

    # Buscar enlace con data-testid="review-summary"
    link = soup.find('a', attrs={'data-testid': 'review-summary'})
    if link and link.get('href'):
        href = link['href']
        if href.startswith('/'):
            return 'https://www.mercadolibre.com.mx' + href
        return href

    return None


def parsear_resenas_pagina(html, product_id, marca, modelo):
    """Extrae reseñas del HTML actual usando BeautifulSoup."""
    soup = BeautifulSoup(html, 'html.parser')
    resenas = []

    articulos = soup.find_all('article', attrs={'data-testid': 'comment-component'})
    if not articulos:
        articulos = soup.find_all('article', class_=re.compile(r'comment'))

    for art in articulos:
        texto_el = art.find('p', class_=re.compile(r'comment__content'))
        fecha_el = art.find('span', class_=re.compile(r'comment__date|comment-date'))
        rating_el = art.find('p', class_='andes-visually-hidden')

        texto = texto_el.get_text(strip=True) if texto_el else ''
        fecha = fecha_el.get_text(strip=True) if fecha_el else ''
        rating = rating_el.get_text(strip=True) if rating_el else ''

        if texto:
            resenas.append({
                'id_review':  '',
                'timestamp':  fecha,
                'raw_text':   texto,
                'clean_text': '',
                'rating':     rating,
                'emotion':    '',
                'intensity':  '',
                'source':     'Mercado Libre',
                'product_id': product_id,
                'category':   '',
                'marca':      marca,
                'modelo':     modelo
            })

    return resenas, soup


def extraer_resenas(producto):
    url        = producto["url"]
    product_id = producto["id"]
    marca      = producto["marca"]
    modelo     = producto["modelo"]
    resenas    = []
    textos_vistos = set()

    try:
        # 1. Ir a la página del producto para encontrar el enlace de reseñas
        driver.get(url)
        time.sleep(5)

        # Scroll para cargar sección de reseñas
        for _ in range(10):
            driver.execute_script("window.scrollBy(0, 600)")
            time.sleep(random.uniform(0.5, 1.0))
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        url_resenas = obtener_url_resenas(soup)

        if not url_resenas:
            # Fallback: extraer lo que haya en la página principal
            nuevas, _ = parsear_resenas_pagina(driver.page_source, product_id, marca, modelo)
            print(f"  ⚠️ {modelo}: No se encontró enlace de reseñas, extrayendo de página principal")
            print(f"✅ {modelo}: {len(nuevas)} reseñas extraídas")
            return nuevas

        print(f"  🔗 {modelo}: Navegando a página de reseñas...")

        # 2. Navegar a la página dedicada de reseñas (usa scroll infinito)
        driver.get(url_resenas)
        time.sleep(random.uniform(3, 5))

        # Scroll infinito: seguir bajando hasta que no aparezcan más reseñas
        sin_cambio = 0
        while len(resenas) < MAX_RESENAS and sin_cambio < 5:
            # Contar reseñas antes del scroll
            antes = len(driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="comment-component"]'))

            # Scroll agresivo hacia abajo
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(1.5, 2.5))

            # Contar reseñas después del scroll
            despues = len(driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="comment-component"]'))

            if despues > antes:
                sin_cambio = 0
                print(f"  📜 {modelo}: {despues} reseñas cargadas...", end='\r')
            else:
                sin_cambio += 1

        # Extraer todas las reseñas cargadas
        nuevas, soup = parsear_resenas_pagina(driver.page_source, product_id, marca, modelo)

        textos_vistos = set()
        for r in nuevas:
            if r['raw_text'] not in textos_vistos:
                resenas.append(r)
                textos_vistos.add(r['raw_text'])

        print(f"✅ {modelo}: {len(resenas)} reseñas extraídas                    ")

    except Exception as e:
        print(f"❌ Error en {modelo}: {e}")

    return resenas


# ─── Extracción ────────────────────────────────────────────────────────
todas = []
for p in productos:
    todas.extend(extraer_resenas(p))
    time.sleep(random.uniform(3, 6))  # pausa entre productos

driver.quit()

# ─── Exportación ───────────────────────────────────────────────────────
df = pd.DataFrame(todas)

if not df.empty:
    df['id_review'] = ['SHOP_' + str(i+1).zfill(4) for i in range(len(df))]
    df.to_csv('raw_laptops_ml.csv', index=False, encoding='utf-8')
    print(f"\n📦 Total reseñas recolectadas: {len(df)}")
    print(f"📁 Archivo guardado: raw_laptops_ml.csv")
else:
    print("\n⚠️ No se recolectaron reseñas.")
    print("Verifica: 1) Las URLs son correctas  2) ML no bloqueó la sesión")

