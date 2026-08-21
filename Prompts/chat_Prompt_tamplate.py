from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

ll, = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7, max_new_tokens=500,
    task="text-generation")

model = ChatHuggingFace(llm=ll) 

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain}assistant."),
    ("user", "{input}")
])  

prompt = chat_prompt.invoke({
    "domain": "research ", 
    "input": "Can you explain the concept of attention mechanism in neural networks?"
})

print(model.invoke(prompt).content)