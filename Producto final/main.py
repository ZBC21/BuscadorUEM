import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Cargar variables de entorno
load_dotenv()

# Configurar API Key de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("API key de OpenAI no configurada.")
    st.stop()

# Modelos
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4.1",
    temperature=0
)

# Cargar el índice FAISS
vectorstore = FAISS.load_local(
    "OpenAI_embedding_large",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Interfaz de usuario Streamlit
st.set_page_config(page_title="Chatbot Universidad", page_icon="UEM.png")

# Imagen y título
col1, col2 = st.columns([1, 7])
with col1:
    st.image("UEM.png", width=120)
with col2:
    st.title(" Asistente Para Búsqueda de Información en la Universidad Europea")

# Inicializar historial en sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Entrada de usuario
query = st.chat_input("Haz tu pregunta relacionada con la universidad...")

# Procesar la consulta
if query:
    with st.spinner("Pensando..."):
        result = qa_chain.invoke({"question": query})
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", result["answer"]))

# Mostrar historial de conversación
for role, text in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Mostrar documentos fuente si existe una respuesta
if "result" in locals():
    with st.expander("📄 Documentos utilizados"):
        for doc in result["source_documents"]:
            st.markdown(f"- **{doc.metadata.get('source', 'Desconocido')}**")

#streamlit run main.py --server.headless true para usarlo
