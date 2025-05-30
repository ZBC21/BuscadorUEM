import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

visited_urls = set()

# Lista de User-Agents para evitar bloqueos
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

# Crear una sesión para reutilizar la conexión y evitar bloqueos
session = requests.Session()

def clean_text(text):
    """Limpia el texto eliminando contenido irrelevante y espacios adicionales."""
    text = re.sub(r'\s+', ' ', text)  # Reemplaza múltiples espacios por uno
    text = re.sub(r'[^\w\s,.!?¿¡]', '', text)  # Elimina caracteres especiales innecesarios
    return text.strip()

def process_natural_language(text):
    """Procesa el texto en lenguaje natural."""
    return text

def crawl_page(url, max_retries=5):
    """Extrae información de una página web con reintentos en caso de error."""
    if url in visited_urls:
        return None

    retries = 0
    while retries < max_retries:
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = session.get(url, headers=headers, timeout=10)

            if response.status_code == 429:  # Too Many Requests
                wait_time = (2 ** retries) + random.uniform(5, 10)  # Exponential backoff Jitter
                time.sleep(wait_time)
                retries += 1
                continue

            response.raise_for_status()  # Lanza excepción si hay error

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else 'Sin título'

            # Eliminar etiquetas no deseadas
            for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'form']):
                tag.decompose()

            text = soup.get_text(separator=' ')  # Asegura que haya espacios entre elementos
            processed_content = process_natural_language(clean_text(text))

            visited_urls.add(url)

            return {
                'title': title,
                'url': url,
                'processed_content': processed_content
            }

        except requests.Timeout:
            pass
        except requests.ConnectionError:
            pass
        except requests.RequestException:
            pass

        retries += 1
        wait_time = (2 ** retries) + random.uniform(3, 10)  # Exponential backoff
        time.sleep(wait_time)

    return None

def crawl_from_index(index_file, output_file):
    """Lee URLs desde un archivo de índice y guarda la información extraída."""
    try:
        with open(index_file, 'r') as f:
            urls = [url.strip() for url in f.readlines() if url.strip()]

        results = []
        failed = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(crawl_page, url): url for url in urls}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"📁 Datos guardados en {output_file}")
        print(f"✅ URLs exitosas: {len(results)} | ❌ URLs fallidas: {failed}")
    except FileNotFoundError:
        print(f"❌ El archivo {index_file} no existe.")

if __name__ == "__main__":
    index_file = "indexSinProfeSinBlog.txt"
    output_file = "output.json"
    crawl_from_index(index_file, output_file)
