
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))

for chunk in chunks:
    print("Chunk content:", repr(chunk.page_content))

# 3. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create the FAISS vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Save the vector store locally
vector_store.save_local("faiss_index")

print("FAISS vector store created and saved!")

# 6. Test semantic search
query = "How do computers learn from data?"
results = vector_store.similarity_search(query, k=2)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)