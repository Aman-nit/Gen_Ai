from langchain_community.tools import tool



##---------------Tools creation using @tool decorator------------------##
@tool
def multiplication(a:int , b:int) -> int:
    """
    Multiply two numbers and return the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of a and b.
    """
    return a * b

result = multiplication.invoke({"a":5 , "b":10})
print("Multiplication Result:\n\n")
print(result)

# #its also have some capability
# print("Tool Name:", multiplication.name)
# print("Tool Description:", multiplication.description)
# print("Tool Args Schema:", multiplication.args_schema)



#--------------------------- Tools creation using structured tool & pydentic ----------------------------##                     
from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultipyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply.")
    b: int = Field(..., description="The second number to multiply.")



def multiply_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers and return the result.

    Args:
        a (int): The first number to multiply.
        b (int): The second number to multiply.

    Returns:
        int: The product of a and b.
    """
    return a * b

multiplication_tool = StructuredTool(
    name="Multiplication_function",
    description="A tool to multiply two numbers.",
    args_schema=MultipyInput,
    func=multiply_numbers
)


result_pydantic = multiplication_tool.invoke({"a": 7, "b": 3})
print("Multiplication Result using Pydantic:\n\n")
print(result_pydantic)


#---------------------using base tool class---------------------##

from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field


class MultiplicationInput(BaseModel):
    a: int = Field(..., description="The first number to multiply.")
    b: int = Field(..., description="The second number to multiply.")


class MultiplyingTool(BaseTool):
    name: str = "Multiplication_tool"
    description: str = "A tool to multiply two numbers."
    args_schema: type[BaseModel] = MultiplicationInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiplication_tool_instance = MultiplyingTool()

result_base_tool = multiplication_tool_instance.invoke({
    "a": 4,
    "b": 6
})

print("Multiplication Result using BaseTool:\n")
print(result_base_tool)
