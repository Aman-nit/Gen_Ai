from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()  # Load the Hugging Face token from the environment.

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(...,gt= 18, description="The person's age")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate.from_template(
    template  = 'give me the name ,age and  city of a fictional {place} person \n {format_instructions}',
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'chinese'})

print(result)