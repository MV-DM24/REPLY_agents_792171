# tasks/initial_task.py
from crewai import Task
from utils.config import config
AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_initial_task(query):
    return Task(
        description=f"""
Process the following user request: '{query}'.
Context:
queries are about users using the NoiPA portal, and the data is aggregated.
Your steps:
1. Understand the request: store it in the memory file and determine if it requires data analysis, visualization, or both.
2. Identify required context: The user is asking about info regarding users of a portal. Remind the agents that the data is aggregated,
and relevant information about the users is stored in multiple columns (lower and upper boundary) and/or files.
3. Delegate: Route the request appropriately to the Data Analyst agent and potentially the Data Visualizer agent, ensuring they receive ALL necessary context (query, data paths, and analyst results if passing to visualizer).
4. Consolidate: Gather the results from the specialist agent(s) and save them in memory.
5. Respond: Formulate a final, clear response to the user based on the outcomes and considering the context (users asking about employees data).
Always provide the analyst's output. If the visualizer agent is not able to generate a visualization, explain why and provide the analysis results anyway.
""",
    expected_output="""A final, consolidated response in the query language addressed to the user. 
    This response should:
    - always contain the analysis resutls in the same language as the original query
    - include visualizations if the visualizer was able to generate them (if not, only include the analysis results)
    The output MUST be in the same language as the original query.""",

    )