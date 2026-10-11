from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # Load environment variables from .env file

video_id = "vNvCQpGYdLo"
Youtube_api = YouTubeTranscriptApi()

def script_loader(video_id)->str:
    try:
        transcript = Youtube_api.fetch(
            video_id,
            languages=["en"]
        )

        print("Transcript fetched successfully!")

        transcript_text = " ".join(
            snippet.text for snippet in transcript
        )
        return transcript_text

    except IpBlocked:
        print("YouTube blocked requests from this IP.")
        return None

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return None


Script_text = script_loader(video_id)
def chunking(script_text:str)->list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(script_text)
    return chunks

print("Script text chunked successfully!")
chunks = chunking(Script_text)  
print(len(chunks))


def embedding(chunks:list)->FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

print("Embeddings created successfully!")
# print(embedding(chunks).index_to_docstore_id)


def retrieve(vectorstore:FAISS ,query:str):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    Related_chunks = retriever.invoke(query)
    return Related_chunks


print("Retrieval successful!")
# print(retrieve(embedding(chunks), "how memory works in our brain?"))

def augmented_query(query:str, related_chunks:list)->str:
    prompt_template = PromptTemplate(
        template = """"You are a helpful assistant .
          answer only from the provided transcript context .
          if the context is in suficieant , just say  i don't know"

        {context}
        question : {question}""",
        input_variables = ["context", "question"]
    )
    context_text = " ".join([chunk.page_content for chunk in related_chunks])
    return prompt_template.format(context=context_text, question=query)


print("Augmented query created successfully!")
# print(augmented_query("how memory works in our brain?", retrieve(embedding(chunks), "how memory works in our brain?")))


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

Answer = model.invoke(augmented_query("how memory works in our brain?", retrieve(embedding(chunks), "how memory works in our brain?")))
print(Answer.content)