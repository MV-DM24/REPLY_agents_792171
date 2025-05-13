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
from tasks.final_task import create_final_reporting_task
from tasks.analyst_tasks import create_analyst_task
from tasks.visualizer_tasks import create_visualization_code_task
from tools.analysis_tool import DataAnalysisTool
from tools.reporter_tool import reporter_tool



AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

st.set_page_config(page_title="Fantastic Crew Analyzer", layout="wide")
st.title(" fantastic Crew: Mavi, Ale, Eli's crew")
st.markdown("""
Enter your query about the NoiPA portal user data.
The crew will analyze the data, prepare a visualization blueprint if requested,
and present the findings directly on this page.
""")

# Initialize session state for storing results if needed (optional, for more complex UIs)
if 'crew_result' not in st.session_state:
    st.session_state.crew_result = None
if 'query_processed' not in st.session_state:
    st.session_state.query_processed = False

# 2. Input user query
with st.form("query_form"):
    query = st.text_input("Enter your data analysis query:", key="user_query_input")
    submit_button = st.form_submit_button("Let's goooo!")

if submit_button and query:
    st.session_state.query_processed = True
    st.session_state.crew_result = None 

    # --- LLM Initialization ---
    try:
        common_llm = ChatLiteLLM(
            model="gemini/gemini-1.5-flash",
            api_key=config.GOOGLE_API_KEY, 
            temperature=0.1, 
            verbose=True 
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM. Please check your API key and configuration: {e}")
        st.stop() # Stop execution if LLM fails


    # --- 3. Instantiate Agents (Pass LLM and Tools) ---
    try:
        analyst_agent = DataAnalystAgent() 

        visualizer_agent = DataVisualizerAgent() 

        reporter_agent = ReporterAgent()
    except Exception as e:
        st.error(f"Error instantiating agents: {e}")
        st.stop()

    # --- 4. Create Tasks ---
    analyst_output_context_placeholder = "{{analyst_data_processing_task.output}}"
    visualizer_output_context_placeholder = "{{visualization_code_generation_task.output}}"

    # Task for Data Analyst
    analyst_data_processing_task = create_analyst_task(analyst_agent = analyst_agent,query=query)

    # Task for Data Visualizer (to generate code)
    visualization_code_generation_task = create_visualization_code_task(
        visualizer_agent=visualizer_agent,
        user_query_for_visualization=query, 
        analyst_task_output_context_name=analyst_output_context_placeholder
    )
    # Set context: Visualizer task needs output from Analyst task
    visualization_code_generation_task.context = [analyst_data_processing_task]


    # Task for Final Reporter (to use Streamlit tool)
    final_report_rendering_task = create_final_reporting_task(
        reporter_agent=reporter_agent,
        original_user_query=query,
        analyst_findings_context_name=analyst_output_context_placeholder,
        visualizer_json_context_name=visualizer_output_context_placeholder
    )
    # Set context: Reporter task needs output from Analyst AND Visualizer tasks
    final_report_rendering_task.context = [analyst_data_processing_task, visualization_code_generation_task]


    # --- 5. Orchestrate the Crew ---

    crew = Crew(
        agents=[analyst_agent, visualizer_agent, reporter_agent], 
        tasks=[
            analyst_data_processing_task,
            visualization_code_generation_task,
            final_report_rendering_task
        ],
        process=Process.sequential, 
        verbose=1
    )

    # --- 6. Run the crew and display the results ---

    st.markdown("---")
    st.subheader("Crew Processing Log & Final Summary:")
    with st.spinner("The crew is processing your request... Please wait."):
        try:
            result = crew.kickoff()
            st.session_state.crew_result = result 
        except Exception as e:
            st.error(f"An error occurred during crew execution: {e}")
            st.session_state.crew_result = f"Crew execution failed: {e}"

# Display the final textual summary from the reporter agent
if st.session_state.query_processed and st.session_state.crew_result:
    st.markdown("### Final Summary from Reporting Agent:")
    st.write(st.session_state.crew_result)

st.markdown("---")
st.markdown("Built with CrewAI & Streamlit.")