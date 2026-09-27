from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()  # Load the Hugging Face token from the environment.

# Create the Hugging Face language model used to generate the response.
llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# Add the parser's JSON-format instructions to the prompt automatically.
template  = PromptTemplate.from_template(
    template = 'Give me the name , age and a city of a frickly person in 5 lines\n {format_instructions}',
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt = template.format()
print(prompt)

# Connect the prompt, model, and parser into one reusable pipeline.
chain = template | model | parser

# Generate a response, then convert its text into a Python object.
result = model.invoke(prompt)
final_result = parser.parse(result.content)  
print(result)
print(final_result)