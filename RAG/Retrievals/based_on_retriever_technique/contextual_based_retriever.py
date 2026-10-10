"""A complete contextual-based retriever example.

The source documents intentionally contain mixed topics.  They are split into
small context chunks first, then an embedding compressor keeps only the chunks
relevant to the user's question.
"""

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_retriever() -> ContextualCompressionRetriever:
    """Create a retriever that removes irrelevant mixed-context chunks."""
    mixed_documents = [
        Document(
            page_content=(
                "Python lists preserve insertion order and allow duplicate values. "
                "They are useful for storing a sequence of items.\n\n"
                "The Amazon rainforest produces oxygen and supports many species.\n\n"
                "A list comprehension provides a compact way to create a list from "
                "an iterable, for example [number * 2 for number in numbers]."
            ),
            metadata={"source": "python-and-nature-notes"},
        ),
        Document(
            page_content=(
                "Python dictionaries store key-value pairs and provide fast average "
                "lookup by key.\n\n"
                "The Pacific Ocean is the largest ocean on Earth.\n\n"
                "Use the dict.get method when a missing key should return a default "
                "value instead of raising KeyError."
            ),
            metadata={"source": "python-and-geography-notes"},
        ),
        Document(
            page_content=(
                "Photosynthesis converts light energy into chemical energy in plants. "
                "This process commonly uses carbon dioxide and water.\n\n"
                "Python functions are reusable blocks of code defined with def."
            ),
            metadata={"source": "science-and-python-notes"},
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    context_chunks = splitter.split_documents(mixed_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(context_chunks, embeddings)
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        k=4,
    )
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )


def main() -> None:
    retriever = build_retriever()
    query = "How do I create a list with a Python list comprehension?"
    results = retriever.invoke(query)

    print(f"Query: {query}")
    print("Relevant context:")
    for index, document in enumerate(results, start=1):
        print(f"Result {index} ({document.metadata['source']}):")
        print(document.page_content)
        print("-" * 60)


if __name__ == "__main__":
    main()