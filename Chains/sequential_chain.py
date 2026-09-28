from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file

#DEFINING THE LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")
LLM = ChatHuggingFace(llm=llm)

template1 = PromptTemplate.from_template(
    template = 'Write a detailed report on the following topic {topic} in 200 words'
    )

template2 = PromptTemplate.from_template(
    template = 'Write a summary of the following report {report} in 5 lines'
    )


parser = StrOutputParser()

chain = template1 | LLM | parser | template2 | LLM | parser



result = chain.invoke({'topic': 'Attention is all you need research paper'})

print("Detailed Report:\n", result)