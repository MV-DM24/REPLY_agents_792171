# tasks/visualizer_tasks.py
from crewai import Task
from utils.config import config
from tools.visualization_tool import visualization_tool
import os

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS


def create_visualization_task(query,):
    return Task(
        description=f"""
        ONLY activated IF the user ask for a visualization.
        Create visualizations using python libraries based on the analysis of the query:
        
        Query: {query}
        
        Based on the analysis results from the data analyst, write and execute Python code that creates visualizations of the data analyst agent analysis.
        The code should:
        1. Look at the analyst agent reasoning and load relevant data from one of these file paths: {AVAILABLE_DATA_PATHS}
        2. Process the data according to the analysis already executed or use the processed data from the analysis
        3. Create visualizations using only available columns that best match the analysis results, think about it
        4. If exact data is not available, create a visualization based on the key insights from the analysis
        5. Display the visualizations
        
        IMPORTANT RULES:
        - ONLY output visualizations of data.
        - Examine datasets to find columns that match concepts in the analysis
        - If no dataset contains the necessary data, create a visualization based on the analysis text itself
        - process the data through simple operations (e.g., groupby, sum, mean) to create the visualization
        - Use the 'Python Visualization Executor' tool for all data visualization
        
        If you don't find matching columns, you MUST use columns already present in the datasets that best match the task.
        Don't be space or case sensitive, reason about the BEST match.
        """,
        expected_output="""Visually-appealing graphs representing the results of relevant analysis, 
        created using Python libraries (e.g., Plotly), based on columns that best match the users query.""",
        tools = [visualization_tool]
    )