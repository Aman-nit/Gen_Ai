from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults()

result = search_tool.run("What is the capital of France?")

print("Search Results:\n\n")
print(result)