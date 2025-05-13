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
***ULTRA-CRITICAL JSON STRING FIELD FORMATTING RULES (APPLY TO VALUES OF "python_code_to_generate_figure" AND "data_for_visualization.value" IF IT'S A STRING):***
    A) **ALL JSON string values MUST be enclosed in double quotes (`"`).**
    B) **Within any JSON string value, special characters MUST be escaped as follows:**
        1.  **Double Quote (`"`)**: Must be escaped as **`\\"`** (backslash followed by a quote).
            Example Python code: `message = "Hello, \\"World\\""` becomes JSON string: `"message = \\"Hello, \\\\\\\"World\\\\\\\"\\""`
        2.  **Backslash (`\\`)**: Must be escaped as **`\\\\`** (two backslashes).
            Example Python code: `path = "C:\\\\temp"` becomes JSON string: `"path = \\"C:\\\\\\\\temp\\""`
        3.  **Newline (`\\n`)**: Must be escaped as **`\\\\n`** (backslash followed by 'n').
            Example Python code with newline: `code = "line1\\nline2"` becomes JSON string: `"code = \\"line1\\\\nline2\\""`
        4.  **Carriage Return (`\\r`)**: Must be escaped as **`\\\\r`**.
        5.  **Tab (`\\t`)**: Must be escaped as **`\\\\t`**.
        6.  **Backspace (`\\b`)**: Must be escaped as **`\\\\b`**.
        7.  **Form Feed (`\\f`)**: Must be escaped as **`\\\\f`**.
    C) **No Unescaped Control Characters:** Ensure no raw control characters (ASCII 0-31) are present within string values unless properly escaped as above.
    D) **Think of it this way:** If you were writing this JSON string in Python, how would you define the string for `python_code_to_generate_figure`?
        Example: `python_code = "import matplotlib.pyplot as plt\\nfig, ax = plt.subplots()\\nax.set_title(\\"My Plot\\")\\nreturn fig"`
        The JSON field would be: `"python_code_to_generate_figure": "import matplotlib.pyplot as plt\\nfig, ax = plt.subplots()\\nax.set_title(\\\"My Plot\\\")\\nreturn fig"`
        Notice how the `\\n` for newline in Python string becomes `\\n` in JSON, and `\\"` for quote in Python string becomes `\\\"` in JSON.
    E) **Double-check the `value` field of `data_for_visualization`:** If `format` is 'csv_string', the `value` will be a single string containing the entire CSV. 
    All newlines within this CSV string must be escaped as `\\\\n`, and any double quotes within the CSV data must be escaped as `\\\"`.

Your adherence to these JSON formatting and escaping rules is ABSOLUTELY ESSENTIAL for the successful completion of your task. 
Double-check your generated JSON string for validity before outputting it.
"""
    expected_output_for_code_visualizer_task = """
A single, well-formatted JSON object adhering to the structures described in the task objective.

A single, raw JSON object string. This string MUST be perfectly parsable by a standard JSON parser.
It MUST NOT be wrapped in markdown ```json ... ``` fences or any other text.

The JSON object MUST conform to one of the two structures (Success Case or Failure Case) detailed in the task description's "Construct Final JSON Output" section (Step 4).

Key requirements for the JSON content:
1.  **Top-level Keys (Success Case):** "visualization_type", "python_code_to_generate_figure", "data_for_visualization", "plot_parameters", "description".
2.  **Top-level Keys (Failure Case):** "visualization_type" (value "none"), "python_code_to_generate_figure" (value null), "data_for_visualization" (value null), "plot_parameters" (value null), "description" (string explaining failure).
3.  **"python_code_to_generate_figure"**: A string containing valid Python code. This code MUST handle parsing of its input data (e.g., if data is a CSV string) and MUST return a Matplotlib Figure object. It MUST NOT call `plt.show()` or `plt.savefig()`.
4.  **"data_for_visualization"**: An object with "format" (string, e.g., 'csv_string') and "value" (the data itself, appropriately typed or serialized as a string).
5.  **JSON Syntax Rules:**
    *   All keys and string values enclosed in **double quotes (`"`)**.
    *   Special characters within any JSON string value (especially within "python_code_to_generate_figure" and "data_for_visualization.value" if it's a string) MUST be correctly escaped:
        *   `"` must be `\\"`
        *   `\\` must be `\\\\`
        *   Newline (`\\n`) must be `\\\\n`
        *   Carriage return (`\\r`) must be `\\\\r`
        *   Tab (`\\t`) must be `\\\\t`
    *   Refer to the "ULTRA-CRITICAL JSON STRING FIELD FORMATTING RULES" in the task description.

Verification: Imagine the output string you generate is directly passed to `json.loads()` in Python. It must parse without any errors.
"""

    return Task(
        description=description_for_code_visualizer_task,
        expected_output=expected_output_for_code_visualizer_task,
        agent=visualizer_agent,
        tools=[]
    )