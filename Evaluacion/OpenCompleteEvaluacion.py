import os
import csv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "Your OpenAi key"
client = OpenAI()
gptModel = "gpt-4.1"

csv_path = "resultados_evaluacion_4_1.csv"
csv_headers = [
    "modelo_embeddings", "query", "respuesta",
    "documentos_fuente", "score_relevancia", "score_calidad"
]

def cargar_queries_desde_txt(path="preguntas_modelo.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

class Score(BaseModel):
    score: int

def evaluar_relevancia_documentos(query: str, documentos: list[str]) -> int:
    prompt = f"""
Actúa como un evaluador que determina la relevancia de los documentos para responder a una consulta. 
Responde siguiendo una escala de Likert entre 1 y 5, donde 1 es que ningún documento es relevante para responder la query y 5 es que todos son relevantes.

# Documentos
{chr(10).join(documentos)}

# Query
{query}
"""
    completion = client.beta.chat.completions.parse(
        model=gptModel,
        messages=[{"role": "user", "content": prompt}],
        response_format=Score,
    )
    return completion.choices[0].message.parsed.score


def evaluar_calidad_respuesta(query: str, respuesta: str) -> int:
    prompt = f"""
Actúa como un evaluador que determina la calidad de la respuesta de un asistente para una consulta.
Responde siguiendo una escala de Likert entre 1 y 5, donde 1 es que la respuesta no es nada adecuada para la consulta y 5 es que la respuesta es muy adecuada para la consulta.

# Consulta
{query}

# Respuesta
{respuesta}
"""
    completion = client.beta.chat.completions.parse(
        model=gptModel,
        messages=[{"role": "user", "content": prompt}],
        response_format=Score,
    )
    return completion.choices[0].message.parsed.score


modelos_para_embeddings = {
    "OpenAI_embedding_large": "text-embedding-3-large"
}

queries = cargar_queries_desde_txt("preguntas_modelo.txt")

with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    for nombre_directorio, nombre_modelo_hf in modelos_para_embeddings.items():
        print(f"\n📊 Evaluando modelo: {nombre_modelo_hf}")
        try:
            embedding = OpenAIEmbeddings(model=nombre_modelo_hf, openai_api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"❌ No se pudo cargar el modelo {nombre_modelo_hf}: {e}")
            continue

        faiss_path = f"../Modelos/OpenAI/{nombre_directorio}"
        if not os.path.exists(faiss_path):
            print(f"⚠️ No se encontró el índice FAISS en: {faiss_path}")
            continue

        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=gptModel, temperature=0),
            retriever=retriever,
            return_source_documents=True
        )

        total_relevancia = 0
        total_calidad = 0
        num_queries = len(queries)

        for query in queries:
            result = chain.invoke({"query": query})
            respuesta = result['result']
            docs = [doc.page_content for doc in result["source_documents"]]

            score_relevancia = evaluar_relevancia_documentos(query, docs)
            score_calidad = evaluar_calidad_respuesta(respuesta, query)

            total_relevancia += score_relevancia
            total_calidad += score_calidad

            print(f"🧠 Query: {query}")
            print(f"📎 Relevancia docs: {score_relevancia} | 💬 Calidad respuesta: {score_calidad}")
            print(f"respuesta: ", respuesta)

            writer.writerow({
                "modelo_embeddings": nombre_modelo_hf,
                "query": query,
                "respuesta": respuesta,
                "documentos_fuente": " ||| ".join(docs),
                "score_relevancia": score_relevancia,
                "score_calidad": score_calidad
            })

        print(f"\n✅ Modelo: {nombre_modelo_hf}")
        print(f"🔹 Relevancia media: {total_relevancia / num_queries:.2f}")
        print(f"🔹 Calidad media: {total_calidad / num_queries:.2f}")

        writer.writerow({
            "modelo_embeddings": nombre_modelo_hf,
            "query": "MEDIA",
            "respuesta": "",
            "documentos_fuente": "",
            "score_relevancia": round(total_relevancia / num_queries, 2),
            "score_calidad": round(total_calidad / num_queries, 2)
        })