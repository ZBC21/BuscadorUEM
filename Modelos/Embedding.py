import json, os
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

FILE_PATH = "output.json"
#"intfloat/multilingual-e5-large",

MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large-instruct",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "embaas/sentence-transformers-multilingual-e5-large",
    "intfloat/e5-base-v2",
    "intfloat/e5-small-v2",
    "intfloat/e5-base",
    "thenlper/gte-base",
    "shibing624/text2vec-base-chinese-paraphrase",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "sdadas/mmlw-e5-small",
    "abhinand/MedEmbed-small-v0.1",
    "intfloat/e5-small",
    "Mihaiii/Ivysaur",
    "avsolatorio/GIST-all-MiniLM-L6-v2",
    "Snowflake/snowflake-arctic-embed-s",
    "Mihaiii/gte-micro-v4",
    "Mihaiii/Bulbasaur",
    "brahmairesearch/slx-v0.1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Mihaiii/Wartortle",
    "minishlab/potion-base-8M",
    "ibm-granite/granite-embedding-30m-english"
]

# 📌 Verificar que el archivo JSON existe y no está vacío
if not os.path.exists(FILE_PATH) or os.stat(FILE_PATH).st_size == 0:
    raise FileNotFoundError("❌ El archivo output.json no existe o está vacío.")

# 📌 Cargar los datos una sola vez
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except json.JSONDecodeError:
    raise ValueError("❌ Error al cargar output.json. Verifica que el JSON esté bien formado.")

# 📌 Procesar documentos en paralelo
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

# 📌 Dividir documentos en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 📌 Extraer texto y metadatos
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

# 📌 Recorrer todos los modelos
for model_name in MODELS:
    print(f"\n🔧 Procesando modelo: {model_name}")
    model_folder = os.path.join("Modelo", model_name.replace("/", "_"))
    os.makedirs(model_folder, exist_ok=True)
    faiss_path = os.path.join(model_folder, "faiss_index")

    # Evitar reentrenamiento si ya existe
    if os.path.exists(faiss_path):
        print("✅ Ya existe el índice, se omite.")
        continue

    try:
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        db = FAISS.from_texts(texts=texts, embedding=embedding_function, metadatas=metadatas)
        db.save_local(faiss_path)
        print(f"✅ Índice FAISS guardado en: {faiss_path}")
    except Exception as e:
        print(f"❌ Error con el modelo {model_name}: {e}")