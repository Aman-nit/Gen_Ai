from langchain_community.tools import ShellTool

shell_tool = ShellTool()

result = shell_tool.invoke("echo Hello, World!")
print("Shell Results:\n\n")
print(result)

"""
others tols can be find https://docs.langchain.com/oss/python/integrations/tools

"""