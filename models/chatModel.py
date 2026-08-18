# from langchain_openai import ChatOpenAI
# from langchain_anthropic import chatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# #-------------------------------this will not work just for demo how to use ---------------------------------
# # Initialize the ChatOpenAI model with your API key and desired parameters
# chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=1500)

# # Invoking the chat model
# result = chat_model.invoke("Write a short story about a robot learning to love.")

# print(result)


# #---------------------------------------Claude api again paid so its just for demo ----------------------------------------------------------------------

# anthropic_chat_model = chatAnthropic(model_name="claude-v1", temperature=0.7, max_tokens=1500)  

# anthropic_result = anthropic_chat_model.invoke("Write a short story about a robot learning to love.")   
# print(anthropic_result)

# #------------------------------------------Google gemini api again paid so its just for demo ---------------------------------------------------------------------- 

# Google_chat_model = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=0.7, max_tokens=1500) 
# Google_result = Google_chat_model.invoke("Write a short story about a robot learning to love.") 
# print(Google_result)

#----------------------------open source Hugging face model ------------------------------- 


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of West Bengal?")

print(result.content)