from langchain_openai import OpenAI
from dotenv import load_dotenv
#loading my api key from the .env file
load_dotenv()

#Initialize the OpenAI LLM with your API key and desired parameters
llm = OpenAI(model_name="gpt-4", temperature=0.7, max_tokens=1500)

#invoking the LLM to generate a short story about a robot learning to love
result = llm.invoke("Write a short story about a robot learning to love.")

print(result)
