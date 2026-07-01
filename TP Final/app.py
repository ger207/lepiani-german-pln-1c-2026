# ─────────────────────────────────────────────────────────────────────────────
# app.py — RAG con HuggingFace Inference API y Gradio
# Para HuggingFace Spaces: configurá el secreto HF_TOKEN en Settings.
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient

# ─── Configuración ────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_IDS = [
    model.strip()
    for model in os.environ.get(
        "HF_MODEL_IDS",
        "HuggingFaceH4/zephyr-7b-beta,mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct"
    ).split(",")
    if model.strip()
]

if not HF_TOKEN:
    raise ValueError("Configurá el secreto HF_TOKEN en el Space.")

# ─── Embeddings ───────────────────────────────────────────────────────────────

modelo_embeddings = SentenceTransformerEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# ─── ChromaDB en memoria ──────────────────────────────────────────────────────

def cargar_paginas_pdf(ruta_pdf):
    reader = PdfReader(ruta_pdf)
    documentos = []
    for numero_pagina, pagina in enumerate(reader.pages):
        texto = pagina.extract_text() or ""
        if texto.strip():
            documentos.append(
                Document(
                    page_content=texto,
                    metadata={"source": str(ruta_pdf), "page": numero_pagina}
                )
            )
    return documentos

vectorstore = Chroma(
    collection_name="proyecto_rag_spaces_v2",
    embedding_function=modelo_embeddings
)

# ─── Divisor de texto ─────────────────────────────────────────────────────────

divisor = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "]
)

# ─── LLM via HuggingFace Serverless Inference ─────────────────────────────────

cliente_hf = InferenceClient(token=HF_TOKEN)

# ─── Pipeline RAG ─────────────────────────────────────────────────────────────

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

TEMPLATE = """Respondé la siguiente pregunta usando ÚNICAMENTE los documentos proporcionados.
Si la respuesta no está, decilo claramente.

Documentos:
{context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]
)

def generar_respuesta_llm(texto_prompt):
    errores = []
    for model_id in MODEL_IDS:
        try:
            respuesta = cliente_hf.chat_completion(
                messages=[{"role": "user", "content": texto_prompt}],
                model=model_id,
                max_tokens=512,
                temperature=0.1,
            )
            return respuesta.choices[0].message.content
        except Exception as exc:
            errores.append(f"{model_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Ningun modelo remoto respondio. " + " | ".join(errores))

def respuesta_extractiva(fragmentos_fuente):
    contexto = formatear_documentos(fragmentos_fuente)
    contexto = contexto[:1800].strip()
    return (
        "No pude conectar con un modelo remoto, "
        "pero estos son los fragmentos mas relevantes:\n\n"
        f"{contexto}"
    )

# ─── Funciones de la interfaz ─────────────────────────────────────────────────

def cargar_pdfs_interfaz(archivos):
    if not archivos:
        return "No se seleccionaron archivos."
    total_paginas = 0
    total_fragmentos = 0
    archivos_ok = []
    errores = []
    batch_size = 128
    for archivo in archivos:
        nombre = Path(archivo.name).name
        try:
            paginas = cargar_paginas_pdf(archivo.name)
            fragmentos = divisor.split_documents(paginas)
            for inicio in range(0, len(fragmentos), batch_size):
                lote = fragmentos[inicio:inicio + batch_size]
                vectorstore.add_documents(lote)
            total_paginas += len(paginas)
            total_fragmentos += len(fragmentos)
            archivos_ok.append(f"- {nombre}: {len(paginas)} paginas, {len(fragmentos)} fragmentos")
        except Exception as exc:
            errores.append(f"- {nombre}: {type(exc).__name__}: {exc}")

    if not archivos_ok and errores:
        return "No se pudo indexar ningun archivo.\n\nErrores:\n" + "\n".join(errores)

    respuesta = [
        "Indexacion completada.",
        f"Archivos indexados: {len(archivos_ok)}",
        f"Paginas indexadas: {total_paginas}",
        f"Fragmentos indexados: {total_fragmentos}",
        "",
        "Detalle:",
        *archivos_ok,
    ]
    if errores:
        respuesta.extend(["", "Archivos con error:", *errores])
    return "\n".join(respuesta)

def responder_pregunta(pregunta, historial):
    historial = historial or []
    if not pregunta.strip():
        return historial, ""
    try:
        fragmentos_fuente = retriever.invoke(pregunta)
        if not fragmentos_fuente:
            respuesta = "Todavia no hay documentos cargados. Primero subi e indexa al menos un PDF."
            historial = historial + [
                {"role": "user", "content": pregunta},
                {"role": "assistant", "content": respuesta},
            ]
            return historial, ""
        texto_prompt = prompt.format(
            context=formatear_documentos(fragmentos_fuente),
            question=pregunta,
        )
        try:
            respuesta = generar_respuesta_llm(texto_prompt)
        except Exception:
            respuesta = respuesta_extractiva(fragmentos_fuente)
    except Exception as exc:
        respuesta = f"Ocurrio un error: {type(exc).__name__}: {exc}"
        historial = historial + [
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": respuesta},
        ]
        return historial, ""

    lineas_fuente = []
    for frag in fragmentos_fuente:
        fuente = Path(frag.metadata.get("source", "desconocida")).name
        pagina = frag.metadata.get("page", "?")
        lineas_fuente.append(f"• {fuente} (pág. {pagina})")
    historial = historial + [
        {"role": "user", "content": pregunta},
        {"role": "assistant", "content": respuesta},
    ]
    return historial, "\n".join(lineas_fuente)

# ─── Interfaz Gradio ──────────────────────────────────────────────────────────

gr.close_all()

with gr.Blocks(title="RAG Local — IFTS24") as demo:
    gr.Markdown("# RAG con HuggingFace Spaces")
    gr.Markdown("**Laboratorio de PLN — IFTS24, 2026**")

    with gr.Tab("📄 Cargar documentos"):
        upload_component = gr.File(
            label="Seleccioná tus PDFs (solo PDF)",
            file_count="multiple"
        )
        boton_cargar = gr.Button("Indexar documentos", variant="primary")
        estado_carga = gr.Textbox(label="Estado", interactive=False, lines=3)
        boton_cargar.click(
            fn=cargar_pdfs_interfaz,
            inputs=[upload_component],
            outputs=[estado_carga]
        )

    with gr.Tab("💬 Hacer preguntas"):
        chatbot_componente = gr.Chatbot(
            label="Conversación",
            height=400,
            type="messages"
        )
        with gr.Row():
            pregunta_componente = gr.Textbox(
                label="Tu pregunta",
                placeholder="¿Qué dice el documento sobre...?",
                scale=4
            )
            boton_preguntar = gr.Button("Preguntar", variant="primary", scale=1)
        fuentes_componente = gr.Textbox(
            label="Fragmentos consultados",
            interactive=False,
            lines=3
        )
        boton_preguntar.click(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )
        pregunta_componente.submit(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )

demo.launch(ssr_mode=False)