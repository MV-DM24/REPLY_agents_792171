# tasks/visualizer_tasks.py
from crewai import Task
from utils.config import config
import os

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS


def create_visualization_code_task(
    visualizer_agent,
    user_query_for_visualization,
    analyst_task_output_context_name
):
    """
    Creates a task for the DataVisualizerAgent to generate Python visualization code and data.
    The agent itself will NOT execute this code or use any tools.

    Args:
        visualizer_agent: The instance of DataVisualizerAgent (expected to have tools=[]).
        user_query_for_visualization (str): The specific part of the user's query that requests a visualization.
        analyst_task_output_context_name (str): A placeholder string (e.g., "{{analyst_task.output}}")
                                                 that CrewAI will replace with the actual output of the
                                                 analyst's task when processing context.
    Returns:
        Task: An instance of a CrewAI Task.
    """
    description_for_code_visualizer_task = f"""
**Objective:** You are ONLY activated IF the user asks for a visualization AND there is structured data from the Data Analyst to generate it. Your goal is to design and generate Python code (using Matplotlib/Seaborn) and the necessary structured data to produce a single, clear visualization. This code and data are intended for later execution by another system (e.g., Streamlit) to render the actual graph. **You will NOT execute any code, save any files, or use any tools.**

**Data Source:**
*   The Data Analyst's structured output (available as: '{analyst_task_output_context_name}') is your EXCLUSIVE data source. Your generated Python code should be prepared to parse/use this data. For example, if '{analyst_task_output_context_name}' provides a CSV string, your Python code must include logic like `import pandas as pd; import io; df_viz_data = pd.read_csv(io.StringIO(analyst_data_string_variable))`.

**Your Steps:**

1.  **Understand Visualization Request & Analyst Data:**
    *   Analyze the user's visualization request: '{user_query_for_visualization}'.
    *   Thoroughly examine the Analyst's structured data from '{analyst_task_output_context_name}'.
    *   If this data is unsuitable or insufficient, your output JSON must clearly explain why no visualization code can be generated.

2.  **Design Visualization & Parameters:**
    *   Choose the SINGLE most appropriate chart type (e.g., bar, line, scatter).
    *   Identify data elements from the Analyst's output for axes, grouping, etc.
    *   Define a chart title, x-axis label, y-axis label, a brief description of the intended visual insight, and the suggested library (e.g., 'matplotlib').

3.  **Generate Python Code & Package Data for Output JSON:**
    *   Write Python code (e.g., Matplotlib/Seaborn) to generate the chart. This code should:
        *   Be designed to accept the necessary data. If the data from '{analyst_task_output_context_name}' is a string (like CSV), your code must include parsing logic.
        *   **NOT include `plt.savefig()` or `plt.show()`.** It should construct the plot object (e.g., a Matplotlib Figure that can be returned by a function in your code).
    *   Prepare the specific data required by your Python code in a structured, JSON-serializable format. Clearly specify the `format` of this data (e.g., 'csv_string', 'json_records_string') and include the `value` in your output JSON.

4.  **Construct Final JSON Output:**
    *   Your final output MUST be a single JSON object containing the Python code, the prepared data, plot parameters, and a description.
    *   **Success Case JSON Structure:**
        ```json
        {{
            "visualization_type": "string (e.g., 'bar_chart')",
            "python_code_to_generate_figure": "string (The Python code to generate the visualization figure. Example: 'import matplotlib.pyplot as plt\\nimport pandas as pd\\nimport io\\ndef generate_plot(data_string):\\n  df = pd.read_csv(io.StringIO(data_string))\\n  fig, ax = plt.subplots()\\n  ax.bar(df[\\'category\\'], df[\\'value\\'])\\n  return fig')",
            "data_for_visualization": {{
                "format": "string (e.g., 'csv_string', 'json_records_string', 'dict_of_lists')",
                "value": "string or dict or list (The actual data, serialized if necessary, that your python_code_to_generate_figure will use)"
            }},
            "plot_parameters": {{
                "title": "string",
                "x_label": "string",
                "y_label": "string",
                "suggested_library": "string (e.g., 'matplotlib')"
            }},
            "description": "string (what the visualization is intended to show)"
        }}
        ```
    *   **Failure Case JSON Structure (if no visualization can be designed):**
        ```json
        {{
            "visualization_type": "none",
            "python_code_to_generate_figure": null,
            "data_for_visualization": null,
            "plot_parameters": null,
            "description": "string (Clear explanation why no visualization code could be generated)"
        }}
        ```
    ***CRITICAL JSON FORMATTING RULES:***
    1.  **Valid JSON Syntax:** Your entire output MUST be a single, valid JSON object. Ensure all keys and string values are enclosed in **double quotes** (e.g., `"myKey": "myValue"`). Do not use single quotes.
    2.  **String Escaping:** When generating the string value for `"python_code_to_generate_figure"` and the string value for `"data_for_visualization.value"` (especially if its format is 'csv_string' or any other string-based format):
        *   All **double quotes (`"`)** within these strings MUST be escaped as **`\\"`**.
        *   All **backslashes (`\\`)** within these strings MUST be escaped as **`\\\\`**.
        *   All **newline characters (`\\n`)** that are part of the content of these strings (e.g., newlines in Python code, newlines in CSV data) MUST be escaped as **`\\\\n`**.
        *   Other special characters like carriage returns (`\\r`) should be `\\\\r`, and tabs (`\\t`) should be `\\\\t`.
        Failure to correctly escape these characters within your string fields will result in an invalid JSON output.
    3.  **No Markdown:** Do not wrap your final JSON output in markdown fences like ```json ... ```. The output must be the raw JSON string itself.

If successful, the JSON MUST contain:
    - "visualization_type": The type of visualization designed (e.g., 'bar_chart').
    - "python_code_to_generate_figure": A string of Python code (e.g., Matplotlib/Seaborn) that will generate the visualization figure. This code MUST NOT save or show the plot itself. It should be self-contained or define a function that can be called with the provided data.
    - "data_for_visualization": An object with a "format" field (e.g., 'csv_string', 'json_records_string') and a "value" field (the actual data string or structure) needed by the "python_code_to_generate_figure".
    - "plot_parameters": A dictionary with "title", "x_label", "y_label", and "suggested_library".
    - "description": A textual description of what the visualization is intended to show.

If unable to design a visualization (e.g., due to unsuitable data), the JSON MUST contain:
    - "visualization_type": "none"
    - "python_code_to_generate_figure": null
    - "data_for_visualization": null
    - "plot_parameters": null
    - "description": A clear explanation of why no visualization code could be generated.

The entire output MUST be a single valid JSON object.
"""
    expected_output_for_code_visualizer_task = """
A single, well-formatted JSON object adhering to the structures described in the task objective.

If successful, the JSON MUST contain these top-level keys:
    - "visualization_type": A string indicating the type of visualization designed (e.g., 'bar_chart', 'line_plot').
    - "python_code_to_generate_figure": A string of Python code (e.g., using Matplotlib/Seaborn) that, when executed, will generate the visualization figure. This code MUST NOT save or show the plot itself. It should be self-contained or define a function that can be called with the provided data. If the data input for the function is a string (e.g., CSV string), the Python code must include the necessary parsing logic (e.g., `import pandas as pd; import io; df = pd.read_csv(io.StringIO(data_string))`).
    - "data_for_visualization": An object containing:
        - "format": A string specifying the format of the data value (e.g., 'csv_string', 'json_records_string', 'json_records_list', 'dict_of_lists').
        - "value": The actual data (e.g., a CSV string, a list of dictionaries, a dictionary of lists) needed by the "python_code_to_generate_figure". This value should be directly usable by the Python code after appropriate parsing if it's a string format.
    - "plot_parameters": A dictionary containing:
        - "title": A string for the chart title.
        - "x_label": A string for the X-axis label.
        - "y_label": A string for the Y-axis label.
        - "suggested_library": A string indicating the primary Python library intended for the code (e.g., 'matplotlib', 'seaborn').
    - "description": A string providing a brief textual description of what the visualization is intended to show and its key insights.

If unable to design a visualization (e.g., due to unsuitable data), the JSON MUST contain these top-level keys:
    - "visualization_type": The string "none".
    - "python_code_to_generate_figure": The value null.
    - "data_for_visualization": The value null.
    - "plot_parameters": The value null.
    - "description": A string clearly explaining why no visualization code could be generated.

The entire output MUST be a single valid JSON object, without any surrounding markdown fences.
All JSON keys and string values must use double quotes.
All special characters within string values (especially in "python_code_to_generate_figure" and "data_for_visualization.value") must be correctly JSON-escaped (e.g., `"` as `\\"`, `\\` as `\\\\`, `\\n` as `\\\\n`).
"""

    return Task(
        description=description_for_code_visualizer_task,
        expected_output=expected_output_for_code_visualizer_task,
        agent=visualizer_agent,
        tools=[]
    )