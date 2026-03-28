import os
import pickle
import faiss
from src.ocr_loader import load_all_documents
from src.rag_engine import prepare_documents, create_index, retrieve, generate_answer
from src.evaluation import evaluate

INDEX_FILE = "vector_store/faiss_index.bin"
DOCS_FILE = "vector_store/documents.pkl"

# Ensure the vector_store directory exists
os.makedirs("vector_store", exist_ok=True)

# --- SAVE & LOAD LOGIC ---
if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    print("Loading saved FAISS index and documents from disk...")
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        documents = pickle.load(f)
else:
    print("First run detected! Loading and processing PDFs (this may take a while)...")
    raw_docs = load_all_documents()
    
    print("Chunking documents...")
    documents = prepare_documents(raw_docs)
    
    print("Creating vector index...")
    index = create_index(documents)
    
    print("Saving index to disk for next time...")
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)

print("\nSystem Ready! ⚖️\n")

# --- CHAT LOOP ---
while True:
    query = input("Ask a legal question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    context, retrieved_docs = retrieve(query, index, documents)

    # Prevent crash if no docs are found
    if not retrieved_docs:
        print("\nAnswer:\nNo relevant documents found in the database.")
        continue

    answer = generate_answer(query, context)

    print("\nAnswer:\n", answer)
    #print("\nSources:\n", context)

    # -------- Evaluation --------
    metrics = evaluate(
        query=query,
        retrieved_docs=[doc["text"] for doc in retrieved_docs],
        answer=answer,
        context=context
    )

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("-" * 80)