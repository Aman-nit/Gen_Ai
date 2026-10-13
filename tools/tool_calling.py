import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv(r"D:\Ai\Lang_Chain\.env", override=True)

# Create a Python tool
@tool
def multiplication(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b

# Connect directly to Groq
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Bind the tool to the model
llm_with_tools = model.bind_tools([multiplication])

# Ask the model
answer = llm_with_tools.invoke("Can you multiply 5 and 6?")

print("AI response:", answer.content)

# Execute the requested tool
for tool_call in answer.tool_calls:
    if tool_call["name"] == "multiplication":
        result = multiplication.invoke(tool_call["args"])
        print("Tool result:", result)