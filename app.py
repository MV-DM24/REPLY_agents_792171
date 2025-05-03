# app.py
import streamlit as st
import os
import sys

# Dynamically determine project root and add to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

#Import Config should be done at top level, not inside IF - Load .env at start
from utils.config import config
from langchain_community.chat_models import ChatLiteLLM

from crewai import Crew, Process
from agents.frontman import FrontmanAgent
from agents.analyst import DataAnalystAgent
from agents.visualizer import DataVisualizerAgent
from tasks.initial_task import create_initial_task
from tasks.analyst_tasks import create_analyst_task
from tasks.visualizer_tasks import create_visualization_task
from tools.analysis_tool import DataAnalysisTool
from tools.visualization_tool import DataVisualizationTool


# 1. Configure Streamlit
st.title("Data Analysis Crew")

# 2. Input user query
query = st.text_input("Enter your data analysis query:")

print(f"Query Value: '{query}'")

if query:
    print("Query is not empty. Proceeding...")  
    llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", api_key=config.GOOGLE_API_KEY, verbose = True)

    frontman_agent = FrontmanAgent()
    analyst_agent = DataAnalystAgent()
    visualizer_agent = DataVisualizerAgent()

    AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS
    initial_task = create_initial_task(query=query)
    analyst_task = create_analyst_task(query=query)
    visualizer_task = create_visualization_task(query=query)


    initial_task.agent = frontman_agent
    analyst_task.agent = analyst_agent
    visualizer_task.agent = visualizer_agent

    crew = Crew(
        agents=[analyst_agent, visualizer_agent],
        tasks=[ analyst_task, visualizer_task],
        process = Process.hierarchical,
        manager_agent = frontman_agent,
        initial_task=initial_task,
        verbose=1,
    )
    # 6. Run the crew and display the results
    st.write("Running the crew...")
    crew_result = crew.kickoff()
    st.write(f"Crew Result: {crew_result}")