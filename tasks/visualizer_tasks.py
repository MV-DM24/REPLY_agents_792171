# tasks/visualizer_tasks.py
from crewai import Task
from utils.config import config
from tools.visualization_tool import visualization_tool
import os

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS


def create_visualization_task(query,):
    return Task(
        description=f"""
ONLY activated IF the user asks for a visualization AND there is data to generate.
Create a clear, single visualization to answer the user's query, using the 'Python Visualization Executor' tool and based on the analyst results (if any).

Steps:
1. Understand the User's Request: Carefully review the original user query to identify the PRIMARY information need.
2. Analyze Available Data:
    - If the Analyst provides data, determine the BEST columns related to the user's request.
    - Prioritize readily available, processed data from the analyst. Load the data with Pandas.
    - If there is no data, clearly say that a visualization cannot be performed.
3. Visualization Design & Code:
    - Choose the single most appropriate and available chart type and always provide chart title and axises label.
    - Create code to answer the user's question.
4. Final Result:
    - Always return code which allows a graph, or a clearly explained reason why no graph could be created, it must make a choice.

RULES:
- Your output MUST be in the format of a JSON object with the following structure:
    {{
        "chart_type": "The type of visualization created",
        "visualization_data": {{"x_values": [...], "y_values": [...], ...}},
        "description": "A description of what the visualization shows",
        "plot_path": "Path to the saved visualization file"
    }}
    
    Ensure the JSON is properly formatted and contains all required fields.
""",
expected_output="""
A properly formatted JSON object containing:
    - chart_type: The type of visualization created
    - visualization_data: Data used for the visualization
    - description: A description of what the visualization shows
    - plot_path: Path to the saved visualization file
""",
        tools = [visualization_tool]
    )