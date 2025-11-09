from langchain_community.tools import DuckDuckGoSearchRun

tool = DuckDuckGoSearchRun()
print(tool.invoke("3i Atlas"))  # Example usage
