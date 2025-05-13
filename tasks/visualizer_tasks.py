# tasks/visualizer_tasks.py
from crewai import Task
from utils.config import config
import os

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS


def create_visualization_task( # Renaming for clarity if you keep both task types
    visualizer_agent, # This agent instance should have the PythonPlottingTool
    user_query_for_visualization: str,
    analyst_task_output_context_name: str # Placeholder for analyst's full output string
):
    """
    Creates a task for the DataVisualizerAgent to generate Python code,
    use its tool to execute it and save a plot image, and then output
    a JSON containing the plot_path and metadata.
    """
    description_for_saving_visualizer_task = f"""
**Objective:** Based on the user's request ('{user_query_for_visualization}') and the Data Analyst's output,
your mission is to create a single, clear plot image, save it to a file using your 'Python Plotting Tool',
and then output a JSON containing the file path and details about the plot.

**Activation:** You are ONLY activated if visualization is requested by the user AND the Data Analyst provides suitable CSV data.

**Data Source & Preparation:**
*   The Data Analyst's full output is available as: '{analyst_task_output_context_name}'.
*   You MUST locate and extract the CSV data string that appears after the '=== DATA FOR VISUALIZATION (CSV) ===' delimiter within the Analyst's output. This extracted CSV string will be referred to as `analyst_csv_data_string`.
*   If this `analyst_csv_data_string` is missing, or if the data it contains is unsuitable for the requested visualization (even after considering adaptations), your output MUST be the 'Failure Case JSON Structure' described below, with a clear explanation in its "description" field.

**Your Workflow:**

1.  **Understand Request & Extract Data:**
    *   Analyze the user's visualization request: '{user_query_for_visualization}'.
    *   Extract the `analyst_csv_data_string` from '{analyst_task_output_context_name}'.
    *   If data extraction fails or data is unsuitable, prepare to output the Failure Case JSON and skip to Step 5.

2.  **Design Visualization & Parameters:**
    *   Based on the user's request and the (parsed) `analyst_csv_data_string`, choose the SINGLE most appropriate chart type (e.g., bar, line, scatter).
    *   Define a clear chart title, x-axis label, and y-axis label. Also, formulate a brief textual description of what the visualization will show (including any adaptations made if the exact request couldn't be met with the data).

3.  **Generate Python Code for Your 'Python Plotting Tool':**
    *   Write Python code (using Matplotlib or Seaborn). This Python code string is what your tool will execute.
    *   This code MUST:
        a.  Expect two variables to be available in its execution scope (provided by the 'Python Plotting Tool'):
            i.  `analyst_data_str`: This will be the `analyst_csv_data_string` you extracted in Step 1.
            ii. `plot_path_to_save`: This will be a file path string (e.g., 'plots/some_image.png') where the plot must be saved.
        b.  Include necessary imports (e.g., `import pandas as pd; import io; import matplotlib.pyplot as plt`).
        c.  Parse `analyst_data_str` into a pandas DataFrame (e.g., `df = pd.read_csv(io.StringIO(analyst_data_str))`).
        d.  Use this `df` to generate the chosen chart, applying the title and axis labels you designed.
        e.  **Crucially, save the generated plot to the `plot_path_to_save` using `plt.savefig(plot_path_to_save)`.**
        f.  Include `plt.close(fig)` after saving to free resources.

4.  **Execute Plotting via Tool:**
    *   You MUST use your 'Python Plotting Tool'.
    *   Call the tool, providing it with:
        *   The Python code string you generated in Step 3.
        *   The `analyst_csv_data_string` you extracted in Step 1.
    *   The tool will execute your code. If successful, it will return the actual `plot_path` (string) where the image was saved. If it fails, it will return an error message (string).

5.  **Construct Final JSON Output (Your SOLE output for this task):**
    *   Your entire response MUST be a single string which is a perfectly valid JSON object. No markdown fences.
    *   Based on the result from your 'Python Plotting Tool':
        *   **If the tool returned a file path (Success):**
            ```json
            {{
                "visualization_type": "string (e.g., 'bar_chart')",
                "plot_parameters": {{
                    "title": "string (Your designed title)",
                    "x_label": "string (Your designed X-label)",
                    "y_label": "string (Your designed Y-label)",
                    "suggested_library": "matplotlib"
                }},
                "description": "string (Your designed description, including any adaptations made)",
                "plot_path": "string (The file path returned by the 'Python Plotting Tool')"
            }}
            ```
        *   **If the tool returned an error message, or if data was unsuitable from Step 1 (Failure):**
            ```json
            {{
                "visualization_type": "none",
                "plot_parameters": null,
                "description": "string (Reason for failure, e.g., 'Data from analyst was unsuitable because X.' or 'Python Plotting Tool Error: [tool's error message here]')",
                "plot_path": null
            }}
            ```
    *   Ensure all JSON keys and string values are enclosed in **double quotes**.
    *   Ensure all special characters within JSON string values (e.g., in the "description") are correctly JSON-escaped (e.g., `\\"` for quotes, `\\\\n` for newlines).

**Adaptation Principle:** If the exact requested visualization isn't possible with the `analyst_csv_data_string`, design and generate code for the most relevant alternative plot. Explain this adaptation clearly in your JSON "description" field. Only if no useful plot can be made from the data should you output the Failure Case JSON.
"""

    expected_output_for_saving_visualizer_task = """
A single, raw JSON object string. This string MUST be perfectly parsable by a standard JSON parser.
It MUST NOT be wrapped in markdown ```json ... ``` fences or any other text.

The JSON object MUST conform to one of the two structures (Success Case or Failure Case) detailed in the task description's "Construct Final JSON Output" section (Step 5).

Key requirements for the JSON content:
1.  **Top-level Keys (Success Case):** "visualization_type", "plot_parameters", "description", and "plot_path".
    *   "plot_parameters" MUST be an object containing "title", "x_label", "y_label", and "suggested_library".
    *   "plot_path" MUST be a string representing the file path to the saved plot image (e.g., '.png').
2.  **Top-level Keys (Failure Case):** "visualization_type" (value "none"), "plot_parameters" (value null), "description" (string explaining failure), "plot_path" (value null).
3.  **JSON Syntax Rules:**
    *   All keys and string values enclosed in double quotes (`"`).
    *   Special characters within any JSON string value MUST be correctly JSON-escaped.

Verification: The output JSON string must be directly usable by `json.loads()` in Python.
"""
    return Task(
        description=description_for_saving_visualizer_task,
        expected_output=expected_output_for_saving_visualizer_task,
        agent=visualizer_agent 
    )

