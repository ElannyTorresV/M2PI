"""
build_index.py

Data Pipeline de Indexación del chatbot RAG de FAQs bancarias.

Etapas:
  1. Carga del documento fuente (data/faq_document.txt)
  2. Segmentación en chunks (por párrafos, con fallback a tamaño fijo + overlap)
  3. Generación de embeddings (OpenAI text-embedding-3-small)
  4. Almacenamiento de chunks + embeddings en data/index.json

Uso:
    python src/build_index.py
"""

import os
import json
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCUMENT_PATH = "data/faq_document.txt"
INDEX_OUTPUT_PATH = "data/index.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Parámetros de chunking (justificados en el README)
CHUNK_MIN_TOKENS = 50
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Cuenta tokens de un texto usando el encoding cl100k_base."""
    return len(_encoder.encode(text))


def load_document(path):
    """Carga el documento fuente en texto plano, manejando encoding UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_paragraphs(text):
    """Divide el documento en párrafos, descartando encabezados de sección (=== ... ===)."""
    raw_paragraphs = text.split("\n\n")
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if not p or p.startswith("==="):
            continue
        paragraphs.append(p)
    return paragraphs


def split_long_paragraph(text):
    """Fallback: divide un párrafo que excede CHUNK_MAX_TOKENS en sub-chunks
    de tamaño fijo (en tokens) con overlap, para no perder contexto en el corte."""
    tokens = _encoder.encode(text)
    sub_chunks = []
    start = 0
    step = CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS
    while start < len(tokens):
        window = tokens[start:start + CHUNK_MAX_TOKENS]
        sub_chunks.append(_encoder.decode(window))
        start += step
    return sub_chunks


def build_chunks(paragraphs):
    """Estrategia de chunking: por párrafos, agrupando párrafos pequeños hasta
    alcanzar CHUNK_MIN_TOKENS, y aplicando split_long_paragraph como fallback
    cuando un párrafo (o agrupación) supera CHUNK_MAX_TOKENS."""
    chunks = []
    buffer_text = ""
    buffer_tokens = 0

    def flush_buffer():
        nonlocal buffer_text, buffer_tokens
        if buffer_text:
            chunks.append(buffer_text.strip())
        buffer_text, buffer_tokens = "", 0

    for paragraph in paragraphs:
        p_tokens = count_tokens(paragraph)

        if p_tokens > CHUNK_MAX_TOKENS:
            flush_buffer()
            chunks.extend(split_long_paragraph(paragraph))
            continue

        if buffer_tokens + p_tokens > CHUNK_MAX_TOKENS:
            flush_buffer()

        buffer_text += ("\n\n" if buffer_text else "") + paragraph
        buffer_tokens += p_tokens

        if buffer_tokens >= CHUNK_MIN_TOKENS:
            flush_buffer()

    flush_buffer()

    return [
        {"chunk_id": f"chunk_{i:03d}", "text": text, "token_count": count_tokens(text)}
        for i, text in enumerate(chunks)
    ]


def generate_embeddings(chunks, client):
    """Genera un embedding por cada chunk usando la API de OpenAI y lo adjunta
    a cada diccionario de chunk bajo la clave 'embedding'."""
    texts = [c["text"] for c in chunks]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    for chunk, item in zip(chunks, response.data):
        chunk["embedding"] = item.embedding
    return chunks


def save_index(chunks, path):
    """Guarda chunks + embeddings en un archivo JSON (vector store simple en disco)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": EMBEDDING_MODEL, "chunks": chunks}, f, ensure_ascii=False)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró OPENAI_API_KEY. Define la variable de entorno "
            "(ver .env.example) antes de ejecutar este script."
        )
    client = OpenAI(api_key=api_key)

    print(f"Cargando documento desde {DOCUMENT_PATH}...")
    text = load_document(DOCUMENT_PATH)

    print("Segmentando en chunks (por párrafos + fallback de tamaño fijo)...")
    paragraphs = extract_paragraphs(text)
    chunks = build_chunks(paragraphs)
    print(f"  -> {len(chunks)} chunks generados "
          f"(tokens min={min(c['token_count'] for c in chunks)}, "
          f"max={max(c['token_count'] for c in chunks)})")

    print(f"Generando embeddings con {EMBEDDING_MODEL}...")
    chunks = generate_embeddings(chunks, client)

    print(f"Guardando índice en {INDEX_OUTPUT_PATH}...")
    save_index(chunks, INDEX_OUTPUT_PATH)

    print("Listo. Índice construido correctamente.")


if __name__ == "__main__":
    main()