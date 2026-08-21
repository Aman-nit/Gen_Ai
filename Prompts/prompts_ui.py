from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate ,load_prompt
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

#backedn invoking the llm 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation")

model = ChatHuggingFace(llm = llm)



#ui of the chat boat
st.header("Research Assistant Chatbot")
st.subheader("Ask me anything about research and I will try to help you!")


#Taking input from user about the responce they want from Our model
paper_type =st.selectbox("Select research paper Name", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])
Style_input = st.selectbox("Select the style of the answer", ["Summary", "Detailed Explanation", "Step-by-step Guide"]) 
output_length = st.slider("Select the output length", min_value=50, max_value=500, value=150, step=10)

#loading Prompt template from the json file
template = load_prompt("Prompts/prompt_template.json")  



#filling the place holders in the prompt with user input
prompt = template.invoke({
    "paper_type": paper_type,
    "style_input": Style_input,
    "output_length": output_length
})


if st.button("Generate Explanation"):
    result = model.invoke(prompt).content
    st.write(result)
