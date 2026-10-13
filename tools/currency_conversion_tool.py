"""
This is a current time currency conversion tool that can get the the current rate from apiu and then calculate teh rate and ansewer the user question.
"""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import os
import requests
import json
from langchain_core.tools import InjectedToolArg
from typing import Annotated


load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

#creating tools for currency conversion factor fetch from api 
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """Get the current exchange rate between two currencies."""

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{os.getenv('EXCHANGE_RATE_API_KEY')}/pair/"
        f"{base_currency}/{target_currency}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("result") != "success":
        raise ValueError(f"Currency API error: {data}")

    return {
        "base_currency": base_currency,
        "target_currency": target_currency,
        "conversion_rate": data["conversion_rate"]
    }

#creating a tool for currency conversion using the conversion factor
@tool
def convert_currency(base_currency_value: float, conversion_rate:Annotated[float, InjectedToolArg]) -> float:
    """Convert a value from one currency to another using the given conversion rate."""
    return base_currency_value * conversion_rate

#Binding tools with LLM 
llm_with_tools = model.bind_tools([get_conversion_factor, convert_currency])

messages = [
    HumanMessage(
        content="What is the current conversion rate from USD to INR "
                "and convert 100 USD to INR?"
    )
]


conversion_factor_result = None

while True:
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print("AI response:", ai_message.content)
        break

    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            conversion_factor_result = get_conversion_factor.invoke(
                tool_call["args"]
            )
            print("Conversion Factor Result:", conversion_factor_result)

            messages.append(
                ToolMessage(
                    content=json.dumps(conversion_factor_result),
                    tool_call_id=tool_call["id"],
                )
            )
        elif tool_call["name"] == "convert_currency":
            if conversion_factor_result is None:
                raise RuntimeError(
                    "convert_currency was requested before the conversion factor "
                    "was available."
                )

            converted_value_result = convert_currency.invoke(
                {
                    **tool_call["args"],
                    "conversion_rate": conversion_factor_result["conversion_rate"],
                }
            )
            print("Converted Value Result:", converted_value_result)

            messages.append(
                ToolMessage(
                    content=str(converted_value_result),
                    tool_call_id=tool_call["id"],
                )
            )
