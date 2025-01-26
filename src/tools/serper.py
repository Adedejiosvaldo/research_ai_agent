from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
import os

class SerperSearchTool:
    def __init__(self):
        self.search = GoogleSerperAPIWrapper()

    def get_tool(self):
        return Tool(
            name="Serper Search",
            func=self.search.run,
            description="Search the internet for recent market information and news"
        )
