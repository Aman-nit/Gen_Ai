from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

pdf_directory = Path(__file__).parent / "books"
loader = DirectoryLoader(
    path=str(pdf_directory),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(docs[2].page_content[:500])  # Print the first 500 characters of the third document's content
print(docs[2].metadata)  # Print the metadata of the third document