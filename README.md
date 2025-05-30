
# 🧠 BuscadorUEM: Asistente de Búsqueda para la Universidad Europea

**BuscadorUEM** es un asistente conversacional inteligente diseñado para mejorar el acceso a la información en la web de la Universidad Europea. Sustituye el formulario tradicional por una interfaz de diálogo que permite realizar consultas complejas en lenguaje natural. Emplea técnicas modernas de NLP, recuperación aumentada (RAG), embeddings y scraping estructurado para ofrecer respuestas precisas y relevantes.

## 🗂️ Estructura del Proyecto

```
BuscadorUEM/
├── Modelos/               # Módulos de embeddings (OpenAI, locales, etc.)
├── Scrap/                 # Scripts de scraping para recopilar contenido de la UEM
├── Producto final/        # Aplicación final y recursos visuales
├── main.py                # Lógica principal de la app (Streamlit o interfaz)
├── output.json            # Datos procesados tras scraping
└── README.md              # Este archivo
```

## ⚙️ Tecnologías y Herramientas

- **Python** (3.11+)
- **LangChain** para orquestación del pipeline RAG
- **OpenAI API** para embeddings y generación de texto
- **FAISS** para almacenamiento y búsqueda semántica de documentos
- **BeautifulSoup + Requests** para el scraper
- **Streamlit** como interfaz conversacional (visual y funcional)
- **HuggingFace Transformers** para embeddings locales alternativos

## 🚀 Cómo Ejecutarlo

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tuusuario/BuscadorUEM.git
   cd BuscadorUEM
   ```

2. **Crea el entorno virtual e instálalo**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # o .venv\Scripts\activate en Windows
   pip install -r requirements.txt
   ```

3. **Configura tus claves**:
   Crea un archivo `.env` con tu API Key de OpenAI:
   ```
   OPENAI_API_KEY=tu_clave_aquí
   ```

4. **Ejecuta la app**:
   ```bash
   cd Producto\ final/
   streamlit run main.py --server.headless true
   ```

## 🔍 Funcionalidades Clave

- **Scraper Personalizado**: extrae contenido útil del sitio web de la UEM.
- **Fragmentación y Embeddings**: convierte texto en vectores semánticos.
- **Recuperación Aumentada**: busca información relevante antes de generar respuestas.
- **Interfaz Amigable**: permite a estudiantes y visitantes hacer preguntas directamente.

## 🎯 Objetivo del Proyecto

Este proyecto tiene como finalidad ofrecer una forma intuitiva, rápida y educativa de encontrar información relevante en el entorno universitario, mejorando la accesibilidad y experiencia del usuario.

## 📄 Licencia

Este proyecto se desarrolla como parte de un Trabajo de Fin de Grado en la Universidad Europea. No está destinado a distribución comercial sin autorización previa.
