from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough ,RunnableParallel,RunnableSequence



load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_prompt_template = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

joke_explanation_prompt = PromptTemplate(
    template="Explain the joke {joke} in simple words",
    input_variables=["joke"]
)

joke_genrator_chain = RunnableSequence( joke_prompt_template | model | parser)

parallel_chain = RunnableParallel({
    'Joke_passthrough': RunnablePassthrough(),
    'Joke_explanation': joke_explanation_prompt | model | parser

}
)

final_chain = joke_genrator_chain | parallel_chain

result = final_chain.invoke({"topic": "Marriege"})
print("Joke about the topic is: \n")
print(f"{result['Joke_passthrough']} \n")
print("Explanation of the joke is: \n")
print(result['Joke_explanation'])
