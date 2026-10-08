import os

os.environ.setdefault("USER_AGENT", "LangChainWebLoader/1.0")

from langchain_community.document_loaders import WebBaseLoader

URL = "https://www.amazon.in/Audio-Technica-ATH-AVC200-SonicPro-Over-Ear-Headphones/dp/B018OMS9O4?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A1WYWER0W24N8S"

loader = WebBaseLoader(URL)

docs = loader.load()

print(len(docs))  # Print the number of documents loaded
print(docs[0].page_content[:500])  # Print the first 500 characters of the first document's content