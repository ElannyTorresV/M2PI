"""
evaluator.py

Agente Evaluador del chatbot RAG de FAQs bancarias.

Recibe (user_question, system_answer, chunks_related) y devuelve un
diccionario con "score" (0-10) y "reason" (justificación), evaluando:
  - Relevancia de los chunks recuperados respecto a la pregunta.
  - Fidelidad de la respuesta al contexto (evita alucinaciones).
  - Completitud: si la respuesta cubre totalmente la pregunta.

Uso:
    python src/evaluator.py
    (evalúa por defecto los ejemplos de outputs/sample_queries.json)
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SAMPLE_QUERIES_PATH = "outputs/sample_queries.json"
EVAL_OUTPUT_PATH = "outputs/evaluation_results.json"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

EVALUATOR_SYSTEM_PROMPT = """Eres un agente evaluador de calidad para un sistema RAG \
de soporte bancario. Evalúa la respuesta generada según tres dimensiones:
1. Relevancia: ¿los chunks recuperados se relacionan con la pregunta del usuario?
2. Fidelidad: ¿la respuesta usa únicamente información presente en los chunks, \
sin inventar datos (alucinaciones)?
3. Completitud: ¿la respuesta cubre totalmente lo que el usuario preguntó?

Devuelve ÚNICAMENTE un JSON válido, sin texto adicional, con esta forma exacta:
{"score": <entero 0-10>, "reason": "<justificación de al menos 50 caracteres, \
mencionando observaciones específicas sobre relevancia, fidelidad y completitud>"}
"""


def load_sample_queries(path):
    """Carga los pares user_question/system_answer/chunks_related a evaluar."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_prompt(question, answer, chunks):
    """Ensambla el prompt que se envía al LLM evaluador, incluyendo la
    pregunta, la respuesta generada y el texto de los chunks usados."""
    chunks_text = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )
    return (
        f"Pregunta del usuario:\n{question}\n\n"
        f"Respuesta del sistema:\n{answer}\n\n"
        f"Chunks recuperados como contexto:\n{chunks_text}"
    )


def evaluate_response(question, answer, chunks, client):
    """Llama al LLM evaluador y devuelve un diccionario {'score', 'reason'}
    validando que el resultado tenga la estructura esperada."""
    user_prompt = build_evaluation_prompt(question, answer, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return validate_evaluation(result)


def validate_evaluation(result):
    """Verifica que el resultado del evaluador cumpla el contrato mínimo:
    score entero 0-10 y reason con al menos 50 caracteres."""
    score = int(result.get("score", -1))
    reason = str(result.get("reason", ""))

    if not (0 <= score <= 10):
        raise ValueError(f"Score fuera de rango: {score}")
    if len(reason) < 50:
        raise ValueError("La justificación (reason) debe tener al menos 50 caracteres")

    return {"score": score, "reason": reason}


def evaluate_all(samples, client):
    """Evalúa una lista de pares consulta-respuesta y adjunta el resultado
    de la evaluación a cada uno."""
    evaluated = []
    for sample in samples:
        evaluation = evaluate_response(
            sample["user_question"],
            sample["system_answer"],
            sample["chunks_related"],
            client,
        )
        evaluated.append({**sample, "evaluation": evaluation})
    return evaluated


def save_evaluations(evaluated, path):
    """Guarda los resultados de evaluación en un archivo JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluated, f, ensure_ascii=False, indent=2)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró OPENAI_API_KEY. Define la variable de entorno "
            "(ver .env.example) antes de ejecutar este script."
        )
    client = OpenAI(api_key=api_key)

    print(f"Cargando ejemplos desde {SAMPLE_QUERIES_PATH}...")
    samples = load_sample_queries(SAMPLE_QUERIES_PATH)

    print(f"Evaluando {len(samples)} respuestas...")
    evaluated = evaluate_all(samples, client)

    for item in evaluated:
        print(f"  - \"{item['user_question'][:50]}...\" -> "
              f"score={item['evaluation']['score']}")

    save_evaluations(evaluated, EVAL_OUTPUT_PATH)
    print(f"Resultados guardados en {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()