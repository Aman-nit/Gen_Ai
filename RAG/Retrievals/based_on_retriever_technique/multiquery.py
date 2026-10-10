
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains.llm import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

# Create simple retriever
simple_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Initialize Hugging Face LLM
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
        task="text-generation"
)

llm = ChatHuggingFace(llm=llm_endpoint)

# Prompt for generating alternative queries
prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "Generate three different versions of the question below "
        "for searching a document database. Return one question per line.\n"
        "Question: {question}"
    )
)

# Create LLM chain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Create Multi-Query Retriever
multi_query_retriever = MultiQueryRetriever(
    retriever=simple_retriever,
    llm_chain=llm_chain,
    include_original=True
)

# Query
query = "How to improve energy levels and maintain balance?"

# Simple retriever results
simple_retriever_results = simple_retriever.invoke(query)

# Multi-query retriever results
multi_query_retriever_results = multi_query_retriever.invoke(query)

print("Simple Retriever Results:\n")
for i, doc in enumerate(simple_retriever_results, start=1):
    print(f"Result {i}:")
    print("Source:", doc.metadata.get("source"))
    print("Content:", doc.page_content)
    print("-" * 60)

print("\nMulti-Query Retriever Results:\n")
for i, doc in enumerate(multi_query_retriever_results, start=1):
    print(f"Result {i}:")
    print("Source:", doc.metadata.get("source"))
    print("Content:", doc.page_content)
    print("-" * 60)
