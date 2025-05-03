# agents/visualizer.py
from crewai import Agent
from utils.config import config
from tools.visualization_tool import visualization_tool
from langchain_community.chat_models import ChatLiteLLM


class DataVisualizerAgent(Agent):
    def __init__(self, llm=None, verbose=True):
        llm = ChatLiteLLM(model="gemini/gemini-1.5-flash",
                          api_key=config.GOOGLE_API_KEY)
        super().__init__(
            role='Data Visualization Expert',
            goal="""
            Create clear, informative, and visually appealing visualizations that accurately represent the key findings and insights provided by the Senior Data Analyst.
            """,
            backstory="""
            You are a talented Data Visualization Specialist. You translate analytical findings and data summaries into compelling visual narratives.
            """,
            verbose=1,
            allow_delegation=False,
            llm=llm,
            tools = [visualization_tool]
        )
