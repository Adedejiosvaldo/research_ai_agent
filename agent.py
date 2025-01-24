

from typing import List, Optional, Dict, Any
from crewai import Agent, Crew, Task, LLM, Process
from pydantic import BaseModel
import logging
import os
import json
from pathlib import Path
from chromadb.config import Settings
import chromadb
from chromadb.types import Collection, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewConfig(BaseModel):
    """Configuration settings for the crew."""
    model_name: str = "gemini/gemini-1.5-pro-latest"
    temperature: float = 0.7
    api_key: Optional[str] = None
    verbose: bool = True

class MarketAnalysisCrew:
    """Manages a crew of AI agents for market analysis."""

    def __init__(self, config: Optional[CrewConfig] = None):
        self.config = config or CrewConfig()
        self._setup_llm()



    def __setup_chroma(self) -> None:
        """Initialize the ChromaDB client."""
        self.chroma_client = chromadb.Client()
        self.results_collection = self.chroma_client.get_or_create_collection("market_analysis_results")



    def _setup_llm(self) -> None:
        """Initialize the LLM with configuration."""
        self.config.api_key = os.getenv("GEMINI_API_KEY")
        if not self.config.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.llm = LLM(
            model=self.config.model_name,
            temperature=self.config.temperature
        )

    def create_agent(self, role: str, goal: str, backstory: str) -> Agent:
        """Create a new agent with specified parameters."""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=self.config.verbose,
            llm=self.llm
        )

    def create_task(self, description: str, expected_output: str, agent: Agent) -> Task:
        """Create a new task with specified parameters."""
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent
        )

    def setup_agents(self) -> List[Agent]:
        """Set up all agents for the crew."""
        return [
            self.create_agent(
                role="Data Analyst",
                goal="Analyze data trends in the market",
                backstory="An experienced data analyst with a background in economics"
            ),
            self.create_agent(
                role="Market Researcher",
                goal="Gather information on market dynamics",
                backstory="A diligent researcher with a keen eye for detail"
            )
        ]

    def setup_tasks(self, agents: List[Agent]) -> List[Task]:
        """Set up all tasks for the crew."""
        return [
            self.create_task(
                description="Collect recent market data and identify trends.",
                expected_output="A report summarizing key trends in the market.",
                agent=agents[0]
            ),
            self.create_task(
                description="Research factors affecting market dynamics.",
                expected_output="An analysis of factors influencing the market.",
                agent=agents[1]
            )
        ]

    def run(self) -> Dict[str, Any]:
        """Execute the crew's tasks and return results."""
        try:
            agents = self.setup_agents()
            tasks = self.setup_tasks(agents)

            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=self.config.verbose
            )

            result = crew.kickoff()
            return self._format_output(result)

        except Exception as e:
            logger.error(f"Error during crew execution: {str(e)}")
            raise

    def _format_output(self, result) -> Dict[str, Any]:
        """Format the crew's output into a structured dictionary."""
        return {
            "raw_output": result.raw,
            "json_output": result.json_dict if hasattr(result, 'json_dict') else None,
            "pydantic_output": str(result.pydantic) if hasattr(result, 'pydantic') else None,
            "tasks_output": result.tasks_output,
            "token_usage": result.token_usage
        }

def main():
    """Main execution function."""
    try:
        crew = MarketAnalysisCrew()
        results = crew.run()

        # Print results in a structured format
        for key, value in results.items():
            if value:
                print(f"\n{key.upper()}:")
                print(json.dumps(value, indent=2) if isinstance(value, dict) else value)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
