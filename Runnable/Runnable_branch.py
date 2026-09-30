from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda,RunnableBranch , RunnablePassthrough


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

def word_counter(report:str):
    return len(report.split())


prompt_genrate_report = PromptTemplate(
    template="Write a detailed  report about {topic}",
    input_variables=["topic"]
)

prompt_genrate_summary = PromptTemplate(
    template="Write a summary about {topic}",
    input_variables=["topic"]
)   

report_genration_chain = RunnableSequence(prompt_genrate_report | model | parser)
summary_genration_chain = RunnableSequence(prompt_genrate_summary | model | parser)

branch_chain = RunnableBranch(
    (lambda x : word_counter(x) > 100 , summary_genration_chain),
    RunnablePassthrough()

)

final_chain = report_genration_chain | branch_chain

result = final_chain.invoke({"topic": "The impact of AI on the future of work"})

print(result)