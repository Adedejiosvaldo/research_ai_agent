from langchain.tools.tavily_search import TavilySearchResults
import os

class TavilyResearchTool:
    def __init__(self):
        self.search = TavilySearchResults(max_results=5)

    def get_tool(self):
        return Tool(
            name="Tavily Research",
            func=self.search.run,
            description="Deep research on specific market topics and companies"
        )
