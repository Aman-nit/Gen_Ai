
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

# 6. retriever 
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5} #lmbda_mult is a parameter that controls the trade-off between relevance and diversity in the retrieved results. A higher value of lambda_mult will prioritize relevance, while a lower value will prioritize diversity.
)

query = "How do computers learn from data?"
results = retriever.invoke(query)
                                      
                                     

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)