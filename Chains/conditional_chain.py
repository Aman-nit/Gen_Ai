"""
we are trying to build an application where we are reviewing our reviews and based on teh sentiment f the review we reply to the review. We are using a conditional chain to do this. The conditional chain will take the sentiment of the review and based on that it will call the appropriate chain to generate a response
"""

from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser  
from pydantic import BaseModel ,Field
from typing import Literal

load_dotenv()  # Load environment variables from .env file

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")
model = ChatHuggingFace(llm=llm)

class feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the review, either 'positive' or 'negative'.")

parser_pydantic = PydanticOutputParser(pydantic_object=feedback)

prompt_Fedback = PromptTemplate.from_template(
    template = 'Classify the following review as positive, negative {review} \n {format_instructions}',
    partial_variables = {'format_instructions': parser_pydantic.get_format_instructions()},

)
prompt_positive = PromptTemplate.from_template(
    template = 'Write a response to the following positive review {feedback} in 2 lines'
    )
prompt_negative = PromptTemplate.from_template(
    template = 'Write a response to the following negative review {feedback} in 2 lines'
    )   

classifier_chain = prompt_Fedback | model | parser_pydantic

# print(classifier_chain.invoke({'review': 'The product is great and I am not satisfied with it'}).sentiment)    

positive_chain = prompt_positive | model | parser
negative_chain = prompt_negative | model | parser


branch_chian =  RunnableBranch(
        (lambda x:x.sentiment == 'positive', positive_chain),
        (lambda x:x.sentiment == 'negative', negative_chain),
        RunnableLambda(lambda x: "The sentiment is neither positive nor negative")

)



final_chain = classifier_chain | branch_chian

final_result = final_chain.invoke({'review': 'The product is  not great and I am not satisfied with it'})

print("Final Result:\n", final_result)