from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)
from dotenv import load_dotenv

load_dotenv()

# Create the Hugging Face language model used to generate the response.
llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

#Defining responce Scheme structure 
schema = [
    ResponseSchema(name = 'fact_1', description = 'fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description = 'fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description = 'fact 3 about the topic'),
]

#Creating aparser 
parser = StructuredOutputParser.from_response_schemas(schema)

#Creating chat Prompt
template = ChatPromptTemplate.from_template(
    template = 'give 3 fact about {topic} \n {format_instruction}',
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)



chain = template |model |parser

result = chain.invoke({'topic': 'black hole'})

print(result)