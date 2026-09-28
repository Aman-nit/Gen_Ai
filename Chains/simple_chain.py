from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

template = PromptTemplate.from_template(
    template = 'write a detailed report on the following topic {topic} in 200 words'
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)

chain.get_graph().print_ascii()
