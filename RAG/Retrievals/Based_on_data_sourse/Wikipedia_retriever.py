
import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# Identify your application to Wikipedia
wikipedia.set_user_agent(
    "LangChainLearning/1.0 (educational project)"
)

query = "Artificial Intelligence"

retriever = WikipediaRetriever(
    lang="en",
    top_k_results=3,
)

try:
    docs = retriever.invoke(query)

    if not docs:
        print("No documents found.")
    else:
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Document {i} ---")
            print("Title:", doc.metadata.get("title", "Unknown"))
            print("Content:", doc.page_content[:500])

except Exception as e:
    print("Wikipedia retrieval failed:", repr(e))
