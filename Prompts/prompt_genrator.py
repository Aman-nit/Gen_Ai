from langchain_core.prompts import PromptTemplate



#creating a prompt based on user input
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_type}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {output_length}

1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with "Insufficient information."
""",
    input_variables=["paper_type", "style_input", "output_length"],
    validate_template=True
)

template.save("Prompts/prompt_template.json")