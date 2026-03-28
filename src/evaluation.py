import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def llm_context_relevance(query, context):
    """Uses the LLM to judge if the retrieved text actually contains the answer."""
    if not context.strip():
        return 0.0
        
    prompt = f"""
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your job is to determine if the provided Context contains enough relevant information to answer the Question.

Question: {query}
Context: {context}

Output ONLY "1" if the context contains the answer or is highly relevant.
Output ONLY "0" if the context is completely irrelevant.
Do not output anything else.
"""
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0} 
        )
        score_text = response['message']['content'].strip()
        # If the LLM output 1, the retrieval was accurate.
        return 1.0 if "1" in score_text else 0.0
    except Exception as e:
        return 0.0

def faithfulness_score(answer, context):
    if "not found in database" in answer.lower():
        return 1.0  
    return cosine_similarity(model.encode([answer]), model.encode([context]))[0][0]

def answer_relevance_score(answer, query):
    if "not found in database" in answer.lower():
        return 0.0 
    return cosine_similarity(model.encode([answer]), model.encode([query]))[0][0]

def evaluate(query, retrieved_docs, answer, context):
    # We drop the fake precision/recall and use real LLM judging
    context_relevance = llm_context_relevance(query, context)
    faithfulness = faithfulness_score(answer, context)
    ans_relevance = answer_relevance_score(answer, query)

    return {
        "Context Relevance (Retrieval)": context_relevance,
        "Faithfulness": round(faithfulness, 3),
        "Answer Relevance": round(ans_relevance, 3)
    }

'''# MUST match the model used in rag_engine.py
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def precision_at_k(retrieved_docs, relevant_docs, k=3):
    retrieved_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in retrieved_k if doc in relevant_docs)
    return relevant_count / k

def recall_at_k(retrieved_docs, relevant_docs, k=3):
    retrieved_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in retrieved_k if doc in relevant_docs)
    return relevant_count / len(relevant_docs) if relevant_docs else 0

def faithfulness_score(answer, context):
    if "not found in database" in answer.lower():
        return 1.0  
    
    return cosine_similarity(
        model.encode([answer]),
        model.encode([context])
    )[0][0]

def relevance_score(answer, query):
    if "not found in database" in answer.lower():
        return 0.0 

    return cosine_similarity(
        model.encode([answer]),
        model.encode([query])
    )[0][0]

def retrieval_accuracy(retrieved_docs, relevant_docs, k=3):
    retrieved_k = retrieved_docs[:k]
    for doc in retrieved_k:
        if doc in relevant_docs:
            return 1  # correct doc found
    return 0  # not found

def evaluate(query, retrieved_docs, relevant_docs, answer, context):
    precision = precision_at_k(retrieved_docs, relevant_docs)
    recall = recall_at_k(retrieved_docs, relevant_docs)
    faithfulness = faithfulness_score(answer, context)
    relevance = relevance_score(answer, query)
    accuracy = retrieval_accuracy(retrieved_docs, relevant_docs)

    return {
        "Retrieval Accuracy": accuracy,
        "Precision@k": round(precision, 3),
        "Recall@k": round(recall, 3),
        "Faithfulness": round(faithfulness, 3),
        "Answer Relevance": round(relevance, 3)
    }'''