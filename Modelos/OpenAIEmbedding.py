import os
import json
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 📌 Cargar API Key desde archivo .env o variable de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ No se encontró la clave de OpenAI. Añádela al entorno o archivo .env.")

# 📌 Configuración del modelo de OpenAI
MODEL_NAME = "text-embedding-3-large"  # O "text-embedding-3-large"

# 📌 Leer JSON de entrada
FILE_PATH = "output.json"
if not os.path.exists(FILE_PATH) or os.stat(FILE_PATH).st_size == 0:
    raise FileNotFoundError("❌ El archivo output.json no existe o está vacío.")

with open(FILE_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 📌 Procesar documentos
def process_item(item):
    if "processed_content" in item and item["processed_content"].strip():
        return Document(
            page_content=item["processed_content"],
            metadata={"source": item["url"], "title": item["title"]}
        )
    return None

with ThreadPoolExecutor() as executor:
    documents = list(filter(None, executor.map(process_item, raw_data)))

if not documents:
    raise ValueError("⚠️ No se encontraron documentos válidos para procesar.")

# 📌 Dividir documentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

# 📌 Carpeta donde se guarda el índice FAISS
save_path = "OpenAI/OpenAI_embedding_large"
os.makedirs(save_path, exist_ok=True)

# 📌 Evitar duplicados
if os.path.exists(os.path.join(save_path, "index.faiss")):
    print("✅ El índice ya existe, se omite.")
else:
    print(f"\n🔧 Usando modelo OpenAI: {MODEL_NAME}")

    # Crear embeddings y guardar en FAISS
    embedding_function = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=openai_api_key)
    db = FAISS.from_texts(texts=texts, embedding=embedding_function, metadatas=metadatas)
    db.save_local(save_path)
    print(f"✅ Índice FAISS guardado en: {save_path}")