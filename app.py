# app.py
import streamlit as st
import os
import sys
import json
import base64

os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"  # Disable telemetry for privacy

# Dynamically determine project root and add to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

#Import Config should be done at top level, not inside IF - Load .env at start
from utils.config import config
from langchain_community.chat_models import ChatLiteLLM

from crewai import Crew, Process
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from agents.reporter import ReporterAgent
from agents.analyst import DataAnalystAgent
from agents.visualizer import DataVisualizerAgent
from tasks.final_task import create_final_task
from tasks.analyst_tasks import create_analyst_task
from tasks.visualizer_tasks import create_visualization_task
from tools.analysis_tool import DataAnalysisTool
from tools.visualization_tool import DataVisualizationTool



AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

# 1. Configure Streamlit
st.title("Fantastic Crew")
st.write("This is a demo of the CrewAI framework, which allows you to create a crew of agents to analyze and visualize data.")


# 2. Input user query
query = st.text_input("Enter your data analysis query:")

if query:
    llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", api_key=config.GOOGLE_API_KEY, verbose=True)
            # 3. Instantiate Agents (Pass LLM)
    frontman_agent = FrontmanAgent()
    analyst_agent = DataAnalystAgent()
    #visualizer_agent = DataVisualizerAgent()


    # 4. Create Tasks
    initial_task = create_initial_task(query=query)
    analyst_task = create_analyst_task(query=query)
    #visualizer_task = create_visualization_task(query=query)



    # 5. Orchestrate the Crew - Set task to agent here instead
    initial_task.agent = frontman_agent
    analyst_task.agent = analyst_agent
    #visualizer_task.agent = visualizer_agent

    crew = Crew(
        agents=[analyst_agent],
        tasks=[initial_task, analyst_task],
        process=Process.hierarchical,
        manager_agent=frontman_agent,
        verbose=1,  
    )

    # 6. Run the crew and display the results
    st.write("Running the crew...")
    result = crew.kickoff()
    st.write(f"{result}")