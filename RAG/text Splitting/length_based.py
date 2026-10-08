from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Ronald Fisher, the English biologist, developed a number of ideas concerning secondary characteristics in his 1930 book The Genetical Theory of Natural Selection, including the concept of Fisherian runaway which postulates that the desire for a characteristic in females combined with that characteristic in males can create a positive feedback loop or runaway where the feature becomes hugely amplified."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0
)

result = splitter.split_text(text)

for i, chunk in enumerate(result):
    print(f"Chunk {i+1}:")
    print(repr(chunk))
    print()