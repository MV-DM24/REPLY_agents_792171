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
""",
expected_output="""
A single, visually-appealing graph representing the results of relevant analysis, created using Python libraries (e.g., Plotly or Matplotlib), using data. If not, a simple reason.
""",
        tools = [visualization_tool]
    )