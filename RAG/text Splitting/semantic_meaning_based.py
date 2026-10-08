from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings=huggingface_embeddings
)

text = """

Machine learning is a branch of artificial intelligence.

It allows computers to learn patterns from data without being explicitly programmed.

Deep learning is a subset of machine learning.

It uses artificial neural networks with multiple layers.

Deep learning is widely used in computer vision and natural language processing.

Natural language processing allows computers to understand human language.

It is used in chatbots, translation systems, and sentiment analysis.

The Indian capital is New Delhi.

India is one of the most populous countries in the world.

The country has a large and diverse population.

Cricket is one of the most popular sports in India.

Many people follow international and domestic cricket tournaments.

"""

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)
    print(chunk)