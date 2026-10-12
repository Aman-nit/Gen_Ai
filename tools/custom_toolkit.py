from langchain_core.tools import tool

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


@tool
def addition(a:int , b:int) -> int:
    """
    Add two numbers and return the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of a and b.
    """
    return a + b


class MathOperations:
    def get_tools(self):
        return [multiplication, addition]


toolkit = MathOperations()
tools = toolkit.get_tools()

for tool in tools:
    print(f"Tool Name: {tool.name}")
