from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

"""
Runnable parallel is a runnable primitive that allows multiple runnables to execute in parallel.It reciev a same inpute and process it parallely and genrate a dictonary of outputs for each runnable. In this example we are using RunnableParallel to genrate a LinkedIn post and a Twitter post in parallel for a given topic.

in this project we are taking a topic from a user and genratting 2 seprate post for posting on LinkedIn and Twitter  using parallel chain 
"""

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt_template_LinkedIn = PromptTemplate(
    template="Write a LinkedIn post about {topic}",
    input_variables=["topic"],
    output_parser=parser
)

prompt_template_Twitter = PromptTemplate(
    template="Write a Twitter post about {topic}",
    input_variables=["topic"],
    output_parser=parser
)

parallel_chain = RunnableParallel(
    {
        "LinkedIn": prompt_template_LinkedIn | model | parser,
        "Twitter": prompt_template_Twitter | model | parser

    }
)

result = parallel_chain.invoke({"topic": "The impact of AI on the future of work"})
print(result['LinkedIn'])
print(result['Twitter'])

