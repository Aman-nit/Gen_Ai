from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

cricketers = [
    "Virat Kohli is an Indian cricketer known for his batting and leadership.",
    "Babar Azam is a Pakistani cricketer famous for his elegant batting style.",
    "Ben Stokes is an English cricketer known for his all-round abilities.",
    "Pat Cummins is an Australian cricketer and a fast bowler.",
    "Kane Williamson is a New Zealand cricketer known for his calm and technically strong batting."
]

query = "tell me about Virat Kohli "

document_embeddings = huggingface_embeddings.embed_documents(cricketers)
query_embedding = huggingface_embeddings.embed_query(query)

# Calculate cosine similarity between the query and each document
scores = cosine_similarity([query_embedding], document_embeddings)

# Get the index of the most similar document
index  , score  = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]


print(f"Query: {query}")
print(f"Most similar document: {cricketers[index]}")
print(f"Similarity score: {score}")