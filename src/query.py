"""
query.py

Query Pipeline del chatbot RAG de FAQs bancarias.

Etapas:
  1. Embedding de la consulta del usuario
  2. Búsqueda vectorial k-NN exhaustivo por similitud coseno
  3. Ensamblado de contexto (chunks recuperados -> prompt)
  4. Generación de la respuesta con un LLM (GPT-4o-mini)

Uso:
    python src/query.py "¿Cómo bloqueo mi tarjeta si la perdí?"
"""

import os
import sys
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INDEX_PATH = "data/index.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = 4  # dentro del rango 2-5 chunks por consulta que pide la rúbrica


def load_index(path):
    """Carga el índice de chunks + embeddings generado por build_index.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def embed_query(question, client):
    """Convierte la pregunta del usuario en un embedding con la misma
    dimensionalidad que los embeddings de los chunks (mismo modelo)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    return np.array(response.data[0].embedding)


def cosine_similarity(vec_a, vec_b):
    """Similitud coseno explícita entre dos vectores: producto punto
    normalizado por la magnitud de cada vector."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def search_similar_chunks(query_vector, chunks, top_k=TOP_K):
    """Búsqueda k-NN exhaustiva: calcula la similitud coseno entre la
    consulta y todos los chunks, y devuelve los top_k más similares.
    A esta escala (decenas de chunks) un k-NN exhaustivo es suficiente
    y más simple/auditable que un índice ANN."""
    scored = []
    for chunk in chunks:
        chunk_vector = np.array(chunk["embedding"])
        score = cosine_similarity(query_vector, chunk_vector)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:top_k]
    return [{"chunk_id": c["chunk_id"], "text": c["text"], "score": float(s)}
            for s, c in top_chunks]


def assemble_context(retrieved_chunks):
    """Formatea los chunks recuperados como contexto numerado para el prompt del LLM."""
    parts = [f"[Fragmento {i+1} | {c['chunk_id']}]\n{c['text']}"
              for i, c in enumerate(retrieved_chunks)]
    return "\n\n".join(parts)


def generate_answer(question, context, client):
    """Genera la respuesta final con el LLM, condicionada estrictamente al
    contexto recuperado (grounding), evitando que invente información."""
    system_prompt = (
        "Eres un asistente de soporte de un banco. Responde la pregunta del "
        "usuario usando EXCLUSIVAMENTE la información del contexto proporcionado. "
        "Si el contexto no contiene la respuesta, indica que no cuentas con esa "
        "información y sugiere contactar a un agente. Sé claro, breve y preciso."
    )
    user_prompt = f"Contexto:\n{context}\n\nPregunta del usuario: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def run_query(question, client, chunks):
    """Orquesta el pipeline completo de consulta y devuelve el JSON estructurado
    con user_question, system_answer y chunks_related."""
    query_vector = embed_query(question, client)
    retrieved = search_similar_chunks(query_vector, chunks, top_k=TOP_K)
    context = assemble_context(retrieved)
    answer = generate_answer(question, context, client)

    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [
            {"chunk_id": c["chunk_id"], "text": c["text"]} for c in retrieved
        ],
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró OPENAI_API_KEY. Define la variable de entorno "
            "(ver .env.example) antes de ejecutar este script."
        )
    if len(sys.argv) < 2:
        print('Uso: python src/query.py "tu pregunta aquí"')
        sys.exit(1)

    question = sys.argv[1]
    client = OpenAI(api_key=api_key)

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            f"No se encontró {INDEX_PATH}. Ejecuta primero: python src/build_index.py"
        )
    chunks = load_index(INDEX_PATH)

    result = run_query(question, client, chunks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()