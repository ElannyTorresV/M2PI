# M2PI
# Chatbot RAG de FAQs — Banco Digital

## Descripción del proyecto

Este proyecto implementa un chatbot de soporte para preguntas frecuentes (FAQ) de un banco, basado en una arquitectura RAG (Retrieval-Augmented Generation). El sistema procesa un documento de texto plano con preguntas y respuestas sobre el aplicativo móvil, las agencias físicas, la página web, seguridad, tarjetas, transferencias, tipos de cuenta, reclamos, comisiones, requisitos y canales de atención. En lugar de depender de búsquedas manuales o de reentrenar un modelo con el conocimiento del banco, el sistema divide el documento en fragmentos (chunks), genera representaciones vectoriales (embeddings) de cada uno, y ante una pregunta del usuario recupera los fragmentos más relevantes por similitud semántica para luego generar una respuesta fundamentada con un LLM. Esto permite responder al instante consultas repetitivas de soporte, con trazabilidad completa de qué fragmentos del documento sustentan cada respuesta.

## Estructura del proyecto

```
rag-faq-banco/
├── data/
│   ├── faq_document.txt      # Documento fuente (~2500 palabras, 36 preguntas de FAQ)
│   └── index.json            # Índice generado por build_index.py (chunks + embeddings)
├── src/
│   ├── build_index.py        # Pipeline de indexación (carga -> chunking -> embeddings -> guardado)
│   ├── query.py               # Pipeline de consulta (embedding -> búsqueda -> generación)
│   └── evaluator.py           # Agente evaluador de calidad de las respuestas
├── outputs/
│   └── sample_queries.json   # 3 ejemplos de consulta-respuesta end-to-end
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación

1. Clona el repositorio e ingresa a la carpeta del proyecto.
   ```
   git clone <url-del-repositorio>
   cd M2PI
   ```
2. Requiere **Python 3.10+**. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Copia `.env.example` a `.env` y define tu API key de OpenAI:
   ```
   cp .env.example .env
   export OPENAI_API_KEY=tu-api-key-aqui
   ```

## Uso

### 1. Construir el índice (pipeline de indexación)

Procesa `data/faq_document.txt`, genera los chunks y sus embeddings, y guarda todo en `data/index.json`:

```
python src/build_index.py
```

Salida esperada (resumen en consola):
```
Cargando documento desde data/faq_document.txt...
Segmentando en chunks (por párrafos + fallback de tamaño fijo)...
  -> 35 chunks generados (tokens min=..., max=...)
Generando embeddings con text-embedding-3-small...
Guardando índice en data/index.json...
Listo. Índice construido correctamente.
```

### 2. Hacer una consulta (pipeline de consulta)

```
python src/query.py "¿Cómo bloqueo mi tarjeta si la perdí?"
```

Salida esperada (formato JSON):
```json
{
  "user_question": "¿Cómo bloqueo mi tarjeta si la perdí?",
  "system_answer": "Debes bloquearla de inmediato desde el aplicativo, en 'Mis tarjetas' > 'Bloquear tarjeta', o llamando a la línea de atención 24 horas...",
  "chunks_related": [
    {"chunk_id": "chunk_019", "text": "¿Qué hago si perdí mi tarjeta o me la robaron? ..."},
    {"chunk_id": "chunk_015", "text": "¿Qué hago si sospecho que fui víctima de un fraude...?"}
  ]
}
```

Puedes ver más ejemplos ya generados en `outputs/sample_queries.json`.

### 3. Evaluar la calidad de una respuesta (agente evaluador)

```
python src/evaluator.py
```

Este script toma los pares de `outputs/sample_queries.json` (o el resultado de una consulta puntual) y devuelve un puntaje de 0 a 10 con una justificación, evaluando relevancia de los chunks, fidelidad de la respuesta al contexto y completitud.

## Decisiones técnicas

**Estrategia de chunking: por párrafos, con fallback a tamaño fijo + overlap.** El documento fuente ya está estructurado como pares de pregunta-respuesta separados por párrafos, así que segmentar por párrafo preserva de forma natural el contexto semántico de cada tema (cada chunk es una unidad de sentido completa). Como red de seguridad, si algún párrafo superara los 500 tokens, se subdivide con una ventana fija de tokens y un overlap de 50 tokens, para no cortar información a la mitad.

**Método de búsqueda vectorial: k-NN exhaustivo con similitud coseno.** Con un documento de este tamaño (35 chunks), calcular la similitud coseno contra todos los vectores es prácticamente instantáneo y evita la complejidad de mantener un índice ANN (como FAISS) que no aporta beneficio real a esta escala. Esto además hace el cálculo totalmente auditable: el score de cada chunk recuperado es explicable con una fórmula simple.

**Modelo de embeddings y generación: OpenAI.** Se usa `text-embedding-3-small` para los embeddings (buena relación calidad/costo) y `gpt-4o-mini` para la generación de respuestas, ambos accedidos con la misma API key. El prompt de generación restringe al modelo a responder solo con la información del contexto recuperado, para minimizar alucinaciones.

**Almacenamiento del índice: JSON plano.** Los embeddings y el texto de cada chunk se guardan en `data/index.json`. Para el volumen de este proyecto (decenas de chunks) esto es suficiente y evita dependencias adicionales de una base de datos vectorial dedicada; migrar a una (por ejemplo Chroma) sería el siguiente paso natural si el corpus de documentos creciera significativamente.