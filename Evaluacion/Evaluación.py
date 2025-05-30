import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.globals import set_debug
set_debug(True)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "Your OpenAi key"
Model_Name="sentence-transformers/paraphrase-mpnet-base-v2"

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """Eres un asistente experto en responder preguntas sobre la Universidad Europea.
Usa la siguiente información recuperada para responder de manera clara y concisa.
Contexto:
{context}
Pregunta:
{question}
Si la información recuperada no es suficiente, responde 'No tengo suficiente información para responder a esto."""
    )
)

embedding_function = HuggingFaceEmbeddings(model_name=Model_Name)
vectorstore = FAISS.load_local("faiss_index_"+Model_Name, embedding_function, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    prompt=custom_prompt
)

def hacer_pregunta(pregunta):
    respuesta = qa_chain.invoke(pregunta)
    print(respuesta)
    if isinstance(respuesta, dict) and "result" in respuesta:
        return respuesta["result"]
    return respuesta

if __name__ == "__main__":
    while True:
        pregunta = input("🗣️ Haz una pregunta (o escribe 'salir' para terminar'): ")
        if pregunta.lower() == "salir":
            print("Saliendo del asistente.")
            break
        respuesta = hacer_pregunta(pregunta)
        print("\n Respuesta del asistente:")
        print(respuesta)
        print("-" * 80)
