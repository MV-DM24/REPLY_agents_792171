# tasks/analyst_tasks.py
from crewai import Task
from utils.config import config
from tools.analysis_tool import analysis_tool

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_analyst_task(query):
    return Task(
        description=f"""Analyze the data based on the provided query: '{query}'.
1. understand the specific information required by the query and go through the memory to check what you already know about the data.
2. systematically use the 'Python Pandas Code Executor' tool to explore the data files available at these paths: {AVAILABLE_DATA_PATHS}.
You MUST start by inspecting all of the datasets to identify which ones could contain the necessary information.
You MUST inspect the columns and sample data (e.g., using df.columns, df.head(), df.info())
of potentially relevant files to determine where the necessary information resides.
3. Understand whether the data is expressed as a category or numberic value, and remember that relevant info may be contained
in columns that don't have the exact name of what you are looking for. If needed, check the column names and if they may contain relevant information to the analysis.
If the data is numeric, check the data type and the range of values.
If the data is categorical, identify the categories and their meanings.
If the available info is available in one dataset, you can use only that one. However, if the required data is spread across multiple files, use all of them and
formulate and execute a plan to merge them using appropriate keys.
4. perform the analysis needed to answer the query '{query}', think about your answer and context and provide a well-reasoned result.
RULES:
- Your output MUST be in the format of a JSON object with the following structure:
    {{
        "summary": "A concise summary of your analysis findings",
        "insights": ["Insight 1", "Insight 2", "Insight 3", ...],
        "metrics": {{"metric_name1": value1, "metric_name2": value2, ...}}
    }}
    
    Ensure the JSON is properly formatted and contains all required fields.
""",
        expected_output=f"""A comprehensive response to the query '{query}',
supported by the analysis performed on the data from {AVAILABLE_DATA_PATHS}.
 A properly formatted JSON object containing:
    - summary: A concise summary of the analysis findings
    - insights: A list of key insights from the data
    - metrics: A dictionary of important metrics calculated during analysis
The response should clearly state the findings. Sometimes the data is aggregated, so you may need to provide a more general response instead of a precise one.
If the analysis is successful, provide clear, data-backed results and concise summaries.
If data exploration or merging was necessary, briefly mention the steps taken.
Always try to give an answer, even stating that some data is general and not precise
only if the query cannot possibly be answered with the available data, provide a clear explanation
detailing which information was missing and which files were checked.
The final response must be in the same language as the original query.""",
        tools=[analysis_tool],
        )

