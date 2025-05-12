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
from utils.guardrail import validate_analysis_output, validate_visualization_output
from utils.callbacks import log_task_completion, track_analysis_completion, track_visualization_completion
from utils.tool_registry import get_tools_for_task
from langchain_community.chat_models import ChatLiteLLM

from crewai import Crew, Process
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from agents.frontman import FrontmanAgent
from agents.analyst import DataAnalystAgent
from agents.visualizer import DataVisualizerAgent
from tasks.initial_task import create_initial_task
from tasks.analyst_tasks import create_analyst_task
from tasks.visualizer_tasks import create_visualization_task
from tools.analysis_tool import DataAnalysisTool
from tools.visualization_tool import DataVisualizationTool


# Define Pydantic models for structured task outputs
class AnalysisOutput(BaseModel):
    summary: str = Field(description="A summary of the analysis findings")
    insights: List[str] = Field(description="Key insights from the data analysis")
    metrics: Dict[str, float] = Field(description="Important metrics calculated during analysis")
    
class VisualizationOutput(BaseModel):
    chart_type: str = Field(description="Type of visualization created")
    visualization_data: Dict[str, Any] = Field(description="Data used for visualization") 
    description: str = Field(description="Description of what the visualization shows")
    plot_path: Optional[str] = Field(description="Path to the saved visualization file if applicable")


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
    visualizer_agent = DataVisualizerAgent()

    # 4. Create Tasks
    initial_task = create_initial_task(query=query)
    analyst_task = create_analyst_task(query=query)
    analyst_task.output_pydantic = AnalysisOutput

    visualizer_task = create_visualization_task(query=query)
    visualizer_task.output_pydantic = VisualizationOutput

    # 5. Orchestrate the Crew - Set task to agent here instead
    initial_task.agent = frontman_agent
    analyst_task.agent = analyst_agent
    visualizer_task.agent = visualizer_agent

    crew = Crew(
        agents=[analyst_agent, visualizer_agent],
        tasks=[initial_task, analyst_task, visualizer_task],
        process=Process.hierarchical,
        manager_agent=frontman_agent,
        verbose=1,  # set verbose here,
    )

    # 6. Run the crew and display the results
    st.write("Running the crew...")
    result = crew.kickoff()

    # 7. Access individual task outputs
    with st.expander("Initial Task Output"):
        st.write(initial_task.output.raw)
    
    with st.expander("Analysis Results"):
        analysis_output = analyst_task.output
        if analysis_output.pydantic:
            st.subheader("Analysis Summary")
            st.write(analysis_output.pydantic.summary)
            
            st.subheader("Key Insights")
            for idx, insight in enumerate(analysis_output.pydantic.insights, 1):
                st.write(f"{idx}. {insight}")
            
            st.subheader("Important Metrics")
            st.json(analysis_output.pydantic.metrics)
        else:
            st.write(analysis_output.raw)  # Fallback to raw output
    
    with st.expander("Visualization Results"):
        viz_output = visualizer_task.output
        if viz_output.pydantic:
            st.subheader(f"{viz_output.pydantic.chart_type} Visualization")
            st.write(viz_output.pydantic.description)
            
            # If there's a saved visualization file, display it
            if viz_output.pydantic.plot_path and os.path.exists(viz_output.pydantic.plot_path):
                try:
                    if viz_output.pydantic.plot_path.endswith('.png') or viz_output.pydantic.plot_path.endswith('.jpg'):
                        st.image(viz_output.pydantic.plot_path)
                    elif viz_output.pydantic.plot_path.endswith('.html'):
                        with open(viz_output.pydantic.plot_path, 'r') as f:
                            html_content = f.read()
                            st.components.v1.html(html_content, height=600)
                except Exception as e:
                    st.error(f"Error displaying visualization: {e}")
        else:
            st.write(viz_output.raw)  # Fallback to raw output






    
    

    