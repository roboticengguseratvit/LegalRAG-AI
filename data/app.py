from src.simple_rag import load_data, create_index, retrieve, generate_answer

print("Loading data...")
docs = load_data()

print("Creating index...")
index, docs = create_index(docs)

print("System Ready!\n")

while True:
    query = input("Ask a legal question (or 'exit'): ")

    if query.lower() == "exit":
        break

    context_list = retrieve(query, index, docs)
    context = "\n".join(context_list)

    answer = generate_answer(query, context)

    print("\nAnswer:\n", answer)
    print("-" * 50)