import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# UPGRADED MODEL: Better semantic understanding for legal text
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def chunk_text(text, chunk_size=1000, overlap=200):
    """Chunks text with a sliding window (overlap) to preserve context."""
    chunks = []
    # Step forward by (chunk_size - overlap)
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def prepare_documents(raw_docs):
    processed_docs = []
    for doc in raw_docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            processed_docs.append({
                "text": chunk,
                "page": doc["page"],
                "source": doc["source"]
            })
    return processed_docs

def create_index(documents):
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, documents, k=3):
    query_vec = model.encode([query])
    # Search for top 10 candidates first, then filter down
    distances, indices = index.search(np.array(query_vec), k * 3)

    results = []
    
    for i, dist in zip(indices[0], distances[0]):
        score = 1 / (1 + dist)  # Convert L2 distance to a 0-1 similarity score
        doc = documents[i]

        # Balanced semantic threshold (fixes overfitting/underfitting)
        # We removed the strict keyword overlap so it understands concepts, not just exact words.
        if score > 0.35: 
            results.append({
                "text": doc["text"],
                "page": doc["page"],
                "source": doc["source"],
                "score": score
            })

    # Fallback: If nothing passes the threshold, grab the closest ones anyway
    if not results:
        for i in indices[0][:k]:
            doc = documents[i]
            results.append({
                "text": doc["text"],
                "page": doc["page"],
                "source": doc["source"],
                "score": 0
            })

    # Sort by highest score and take top K
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    formatted = []
    for doc in results:
        formatted.append(f"[{doc['source']} | Page {doc['page']}]\n{doc['text']}")

    return "\n\n".join(formatted), results

def generate_answer(query, context):
    prompt = f"""
You are a helpful and accurate Indian legal assistant.

Instructions:
1. Answer the question using ONLY the provided Context.
2. You are allowed to summarize, extract, or explain the legal rules found in the Context to answer the user's question.
3. Do not use any outside knowledge.
4. If the exact answer cannot be deduced from the Context, you MUST output the exact string "Not found in database." and NOTHING else. Do not explain yourself. Do not apologize.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0} 
    )

    return response['message']['content']