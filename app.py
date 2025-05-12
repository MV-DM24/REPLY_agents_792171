# app.py
import streamlit as st
import os
import sys
import json
import base64

# Dynamically determine project root and add to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

#Import Config should be done at top level, not inside IF - Load .env at start
from utils.config import config
from langchain_community.chat_models import ChatLiteLLM

from crewai import Crew, Process
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from agents.frontman import FrontmanAgent
from agents.analyst import DataAnalystAgent
from agents.visualizer import DataVisualizerAgent
from tasks.initial_task import create_initial_task
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
    visualizer_agent = DataVisualizerAgent()

    # 4. Create Tasks
    initial_task = create_initial_task(query=query)
    analyst_task = create_analyst_task(query=query)
    visualizer_task = create_visualization_task(query=query)

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
    final_result = crew.kickoff()

import re
import json
from crewai import CrewOutput
from typing import Optional, Tuple

def clean_and_extract_results(crew_output: CrewOutput) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Processes a CrewOutput object to extract and clean analysis and visualization results.
    Prioritizes structured data and handles potential errors for robust result extraction.

    Args:
        crew_output: The CrewOutput object from a CrewAI task execution.

    Returns:
        A tuple containing:
        - analysis_result: The analysis result as a string, or None if not available.
        - visualization_result: The visualization data as a JSON string, or None if not available.
        - error_message: An error message, or None if no error occurred.
    """

    analysis_result = None
    visualization_result = None
    error_message = None

    try:
        # 1. Prioritize Pydantic Output (Structured Data)
        if crew_output.pydantic:
            #Assumed to be all that, then go to this step
            #Since its Pydantic, it won't be just a str, you can create this from string as well
            try:
                #json = crew_output.pydantic.json()
                import json
                # Ensure there is a to-json
                if hasattr(crew_output.pydantic, 'to_json'):
                  # If there's to JSON, then treat it as that.
                  json_result = crew_output.pydantic.to_json()
                  output  = json.loads(json_result) #This means to dump as string is working.
                else:
                  #Treat it as a Str object
                  output = str(crew_output.pydantic)

                if output["result_type"] == "matplotlib" or output["result_type"] == "plotly":
                    visualization_result = output.get("plotly_data") or output.get("image_data")
                elif output["result_type"] == "text":
                    analysis_result = output.get("text_data")
                elif output["result_type"] == "error":
                    error_message = output.get("error_message")
                else:
                    error_message = f"Unknown pydantic.result_type : {str(crew_output.pydantic)}" #The data type should be JSON,

            except Exception as e: #error from load
              error_message = f"Error Pydanctic result but was an error, double chekc! : {str(e)}"
        #If we don't know how to work with the type of input, then skip and
        elif crew_output.raw: # what is this? Can't parse
              output_string = str(crew_output.raw) #if it is it string
              try:
                  crew_output = json.loads(output_string)

                  if output["result_type"] == "matplotlib" or output["result_type"] == "plotly":
                      visualization_result = output.get("plotly_data") or output.get("image_data")
                  elif output["result_type"] == "text":
                      analysis_result = output.get("text_data")
                  elif output["result_type"] == "error":
                      error_message = output.get("error_message")
                  else:
                      error_message = f"Unknown raw result_type: {str(crew_output.raw)}"
              except Exception as e:
                  error_message = f"Had something, not loading error: {str(e)}"
    except Exception as e:
      error_message = f"Has full object as error: {str(e)}"
      
    except Exception as e: #handle that load for everything

                st.error(f"An error occurred during the last processing step. If you are trying to load a graph object there is a code implementation problem that will need to be addressed. Please check the debugging for the current state and ask for help. {e}")
                st.write(error_message)

    return analysis_result, visualization_result, error_message



if final_result:
        analysis_result, visualization_result, error_message = clean_and_extract_results(final_result)

        if error_message:
            st.error(f"Error: {error_message}")
        else:
            if analysis_result:
                st.subheader("Analysis Result")
                st.write(analysis_result)
            if visualization_result:
                st.subheader("Visualization Result")
                st.image(visualization_result)
else:
        st.error("No result returned from the crew.")



    
    

    