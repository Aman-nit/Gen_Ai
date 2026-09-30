from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough,RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt_joke_genrator = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)
runnable_passthrough = RunnablePassthrough()

Joke_genrator_chain = RunnableSequence(prompt_joke_genrator | model | parser)

def word_counter(joke:str):
    return len(joke.split())

Runnable_word_counter = RunnableLambda(word_counter)

runnable_parallel = RunnableParallel({
    'Joke_passthrough': runnable_passthrough,
    'Joke_word_count': Runnable_word_counter
}
)

final_chain = Joke_genrator_chain | runnable_parallel

result = final_chain.invoke({"topic": "AI"})

print("Joke about the topic is: \n")
print(f"{result['Joke_passthrough']} \n")
print("Word count of the joke is: \n")
print(result['Joke_word_count'])