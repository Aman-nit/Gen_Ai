from langchain_openai import  OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

#loading my api key from the .env file
load_dotenv()

#--------------------------------this will not work just for demo how to use ---------------------------------
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimension = 30)

# #here we use embed_query to generate an embedding for the query "What is the capital of West Bengal?"
# result = embeddings.embed_query("What is the capital of West Bengal?")

# documents = ["Kolkata is the capital of West Bengal.", 
#              "The capital of West Bengal is Kolkata.", 
#              "West Bengal's capital city is Kolkata."
#              ]

# #for each document in the list, we generate an embedding using embed_documents
# document_embeddings = embeddings.embed_documents(documents)

# print(result)
# print(document_embeddings)


#-------------------------------------using opensource embedding model it will run -----------------------------------------

huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of West Bengal?"

huggingface_result = huggingface_embeddings.embed_query(text)
print(huggingface_result)