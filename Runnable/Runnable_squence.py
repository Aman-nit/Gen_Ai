from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt_joke = PromptTemplate(
    template = 'write a joke about {topic}'
)
prompt_explanation = PromptTemplate(
    template = 'Explain the following joke {text}'
)

chain = RunnableSequence(prompt_joke , model , parser , prompt_explanation , model, parser)

# result = chain.invoke({'topic':'About machine'})
# print(result)

print(chain.get_graph().draw_ascii())
