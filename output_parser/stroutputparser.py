from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()  # Load environment variables from .env file

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

#1st prompt ->detailed report 
templat1 = PromptTemplate.from_template(
    template = 'write a detailed report on the following topic {topic} in 200 words'
)


#2nd prompt -> summary of the report
templat2 = PromptTemplate.from_template(
    template = 'write a summary of the following report {report} in 5 line '
)                    

prompt = templat1.invoke({'topic': 'black hole'})

result = model.invoke(prompt).context

prompt2 = templat2.invoke({'report': result})

result2 = model.invoke(prompt2).context

print("Detailed Report:\n", result)

print("Summary:\n", result2)



