# tasks/visualizer_tasks.py
from crewai import Task
from utils.config import config
import os

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS


def create_visualization_task( 
    visualizer_agent, 
    user_query_for_visualization: str,
    analyst_task_output_context_name: str 
):
    """
    Creates a task for the DataVisualizerAgent to generate Python code,
    use its tool to execute it and save a plot image, and then output
    a JSON containing the plot_path and metadata.
    """
    description_for_saving_visualizer_task = f"""
**Objective:** You are ONLY activated IF the user asks for a visualization AND there is structured data from the Data Analyst to generate it. Your goal is to design and generate Python code (using Matplotlib/Seaborn) and the necessary structured data to produce a single, clear visualization. This code and data are intended for later execution by another system (e.g., Streamlit) to render the actual graph. **You will NOT execute any code, save any files, or use any tools. Your SOLE output is a single, valid JSON object string.**

**Data Source:**
*   The Data Analyst's structured output (available as: '{analyst_task_output_context_name}') is your EXCLUSIVE data source.
*   You MUST extract the CSV data string that appears after the '=== DATA FOR VISUALIZATION (CSV) ===' delimiter within the Analyst's output. This extracted CSV string will be referred to as `source_csv_data_string`.
*   Your generated Python code for the `"python_code_to_generate_figure"` field must include the necessary import statements (e.g., `import pandas as pd; import io;`) and parsing logic to handle this `source_csv_data_string` (e.g., `df = pd.read_csv(io.StringIO(source_csv_data_string))`). The data variable your Python code uses (e.g., `df`) will effectively become the `df_viz_data` when executed by the downstream tool.

**Your Workflow:**

1.  **Understand Request & Extract/Prepare Data:**
    *   Analyze the user's visualization request: '{user_query_for_visualization}'.
    *   Extract the `source_csv_data_string` from '{analyst_task_output_context_name}'.
    *   This `source_csv_data_string` will be the content for your `"data_for_visualization.value"` field, with `"format": "csv_string"`.
    *   If data extraction fails, or the `source_csv_data_string` is missing or unsuitable for the requested visualization (even after considering adaptations), your output MUST be the 'Failure Case JSON Structure' described below.

2.  **Design Visualization & Parameters:**
    *   Based on the user's request and the (conceptual) DataFrame parsed from `source_csv_data_string`, choose the SINGLE most appropriate chart type.
    *   Identify specific column names from the `source_csv_data_string` that your Python code will use for axes, grouping, etc.
    *   Define a chart title, x-axis label, y-axis label, a brief descriptive summary of the visual insight, and the suggested library (e.g., 'matplotlib').

3.  **Generate Python Visualization Code:**
    *   Write Python code for the `"python_code_to_generate_figure"` field. This code MUST:
        a.  Include all necessary imports (e.g., `import matplotlib.pyplot as plt`, `import pandas as pd`, `import io`).
        b.  **Crucially, it should expect the data to be available via a variable named `df_viz_data`. The tool executing this code will be responsible for creating `df_viz_data` by parsing the `value` from your `data_for_visualization` field based on its `format`. Your Python code should NOT redefine `df_viz_data` from a raw string if the executor already provides it as a DataFrame.**
            Alternatively, and perhaps more robustly, design your Python code to define a function like `def generate_plot(df_data_input, plot_parameters_input): ... return fig`. The executor tool would then call this function. For simplicity, we will assume direct use of `df_viz_data` for now, but a function is better.
        c.  Use `df_viz_data` and `plot_params` (which will also be injected into the execution scope by the executor tool) to construct the plot (e.g., `ax.set_title(plot_params.get('title', 'Default Title'))`).
        d.  Construct a Matplotlib figure and axes (e.g., `fig, ax = plt.subplots()`).
        e.  Perform all plotting operations on `ax`.
        f.  **ABSOLUTELY CRITICAL: The executor tool will provide a variable `plot_path_to_save`. Your Python code MUST save the figure using `plt.savefig(plot_path_to_save)`.**
        g.  After saving, include `plt.close(fig)`.
        h.  Assign the figure to `figure_object = fig`.
        i.  DO NOT include `plt.show()`.

4.  **Construct Final JSON Output - ADHERE STRICTLY TO THE FOLLOWING:**
    *   Your entire response for this task MUST be a SINGLE string which is a **perfectly valid JSON object**. No markdown.
    *   **Success Case JSON Structure:**
        ```json
        {{
            "visualization_type": "bar_chart",
            "python_code_to_generate_figure": "import matplotlib.pyplot as plt\\nimport pandas as pd\\nimport io\\n# Assumes df_viz_data, plot_params, plot_path_to_save, figure_object are in scope\\n# (df_viz_data, plot_params, plot_path_to_save are injected by executor)\\nfig, ax = plt.subplots()\\nax.bar(df_viz_data['X_COLUMN_NAME'], df_viz_data['Y_COLUMN_NAME'])\\nax.set_title(plot_params.get('title', 'Default'))\\nplt.savefig(plot_path_to_save)\\nplt.close(fig)\\nfigure_object = fig",
            "data_for_visualization": {{
                "format": "csv_string",
                "value": "X_COLUMN_NAME,Y_COLUMN_NAME\\nDataA,10\\nDataB,20"
            }},
            "plot_parameters": {{ "title": "...", "x_label": "...", "y_label": "...", "suggested_library": "matplotlib" }},
            "description": "..."
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

The JSON object MUST conform to one of the two structures (Success Case or Failure Case) detailed in the task description.

Key requirements for the JSON content (Success Case):
1.  "visualization_type": string.
2.  "python_code_to_generate_figure": A string of Python code. This code:
    *   MUST expect variables `df_viz_data` (pandas DataFrame), `plot_params` (dict), and `plot_path_to_save` (string) to be available in its execution scope.
    *   MUST use these variables to generate a Matplotlib plot.
    *   MUST call `plt.savefig(plot_path_to_save)`.
    *   MUST assign the created figure to `figure_object = fig`.
    *   MUST NOT call `plt.show()`.
3.  "data_for_visualization": An object with "format" (string, e.g., 'csv_string') and "value" (the actual data string or structure that will be parsed into `df_viz_data` by the executor).
4.  "plot_parameters": A dictionary with "title", "x_label", "y_label", "suggested_library".
5.  "description": A string.

Verification: The output JSON string must be directly usable by `json.loads()` in Python.
"""
    return Task(
        description=description_for_saving_visualizer_task,
        expected_output=expected_output_for_saving_visualizer_task,
        agent=visualizer_agent 
    )

