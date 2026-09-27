from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv 
from langchain_core.prompts import chat_prompt_template ,message_placeholder
load_dotenv()

#chat template 
chat_template = chat_prompt_template([
    ('system', "You are a helpful assistant."),
    message_placeholder(variable_name="user_input"),
    ('human', "User: {user_input}"),]
)
chat_history = []   

#loading chat history 
with open("chat_history.txt", "r") as file:
    chat_history.append(file.readlines())

#creating prompot
chat_template.invoke({
    'chat_history': chat_history,
    'user_input': "where we use it"
})