import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

DATA_PATH = "data"
INDEX_PATH = "vector_store"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_data():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".json"):
            with open(os.path.join(DATA_PATH, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    text = f"Question: {item['question']} Answer: {item['answer']}"
                    docs.append(text)

    return docs


def create_index(docs):
    embeddings = model.encode(docs)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, docs


def retrieve(query, index, docs, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    results = [docs[i] for i in indices[0]]
    return results


def generate_answer(query, context):

    prompt = f"""
You are a legal assistant.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']