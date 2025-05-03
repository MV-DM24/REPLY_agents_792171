# tasks/initial_task.py
from crewai import Task
from utils.config import config
AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_initial_task(query):
    return Task(
        description=f"""
Process the following user request: '{query}'.

Your steps:
1. Understand the request: store it in the memory file and determine if it requires data analysis, visualization, or both.
2. Identify required context: You MUST use and pass the available data paths: {AVAILABLE_DATA_PATHS}.
3. Delegate: Route the request appropriately to the Data Analyst agent and potentially the Data Visualizer agent, ensuring they receive ALL necessary context (query, data paths, and analyst results if passing to visualizer).
4. Consolidate: Gather the results from the specialist agent(s) and save them in memory.
5. Respond: Formulate a final, clear response to the user based on the outcomes and considering the context (users asking about employees data).
If the request is impossible, explain why based on agent feedback.
""",
    expected_output="""A final, consolidated response addressed to the user. 
    This response should either contain the requested analysis summary and/or visualization, 
    or a clear explanation based on specialist agent feedback if the request could not be fulfilled using the provided data files.
    The output MUST be in the same language as the original query.""",

    )