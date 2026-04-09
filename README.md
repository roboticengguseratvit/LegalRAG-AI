# AI-Based Legal Assistant using RAG

An AI-powered legal assistant built using a **Retrieval-Augmented Generation (RAG)** framework to provide accurate, context-aware answers from legal documents. The system works **offline**, ensuring privacy and reliability.

---

## Features

- Supports both **text-based and scanned PDFs**
- Semantic search using **BGE embeddings**
- Fast retrieval with **FAISS vector database**
- Answer generation using **Llama3 (via Ollama)**
- Evaluation using:
  - Context Relevance
  - Faithfulness
  - Answer Relevance
- Interactive UI with **Streamlit**
- Fully **offline system**

---

## How It Works

1. **Document Processing**
   - Extract text using `PyPDF`
   - Use `pytesseract + pdf2image` for scanned PDFs

2. **Preprocessing**
   - Clean text
   - Split into chunks with overlap

3. **Embedding**
   - Convert text into vector embeddings using **BGE model**

4. **Storage**
   - Store embeddings in **FAISS**

5. **Query Processing**
   - Convert user query into embedding
   - Retrieve top relevant chunks

6. **Answer Generation**
   - Use **Llama3 via Ollama** to generate answers

7. **Evaluation**
   - Measure:
     - Context Relevance (LLM-as-a-judge)
     - Faithfulness
     - Answer Relevance

---

## System Architecture

> RAG Pipeline: Document → Embedding → FAISS → Retrieval → LLM → Answer

---

## Tech Stack

- **Python**
- **SentenceTransformers (BGE embeddings)**
- **FAISS**
- **Ollama (Llama3)**
- **PyPDF, pytesseract, pdf2image**
- **Streamlit**
- **scikit-learn, NumPy**

---

## Results

| Metric | Score |
|--------|------|
| Context Relevance | 0.967 |
| Faithfulness | 0.871 |
| Answer Relevance | 0.626 |

---

## Limitations

- Moderate answer relevance (needs better alignment)
- Depends on retrieved context
- No domain-specific fine-tuning
- Limited dataset
- No real-time updates

---

## Future Work

- Improve prompt engineering
- Add re-ranking mechanism
- Use fine-tuned legal LLMs
- Expand dataset
- Add real-time updates

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
