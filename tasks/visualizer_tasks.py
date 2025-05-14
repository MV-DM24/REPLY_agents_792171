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

3.  **Generate Python Code & Package Data for Output JSON:**
    *   Write Python code (e.g., Matplotlib/Seaborn) to generate the chart. This code will be the value for 
        the `"python_code_to_generate_figure"` field in your JSON output.
    *   This Python code MUST:
        a.  Include all necessary imports (e.g., `import matplotlib.pyplot as plt`, `import pandas as pd`, 
            `import io`).
        b.  Be designed to accept the necessary data. If the `data_for_visualization.format` is 'csv_string' 
            (from the Analyst), your code must parse it 
            (e.g., `df = pd.read_csv(io.StringIO(data_string_variable_containing_csv))`). 
            The data will be available in a variable like `df_viz_data` injected by the executing tool.
        c.  Use the `df_viz_data` and `plot_params` (also injected) to construct the plot 
            (e.g., `ax.set_title(plot_params.get('title', 'Default Title'))`).
        d.  **ABSOLUTELY CRUCIAL: After creating the figure (`fig`), you MUST explicitly save it 
            using the `plt.savefig()` command. The file path to save to will be provided to your code 
            in a variable named `plot_path_to_save`. Your code MUST use this exact variable: 
            `plt.savefig(plot_path_to_save)`.**
        e.  After saving, include `plt.close(fig)` to release memory.
        f.  The code should also aim to assign the created Matplotlib figure object to a variable named 
        `figure_object` (e.g., `figure_object = fig`) so it can also be used for direct display by `st.pyplot()`.
    *   **DO NOT include `plt.show()` in your generated code.**
*   Prepare the specific data required by your Python code in the `data_for_visualization` part of your JSON.

4.  **Construct Final JSON Output:**
    *   Your final output MUST be a single JSON object.
    *   **Success Case JSON Structure Example (Illustrative):**
        ```json
        {{
            "visualization_type": "bar_chart",
            "python_code_to_generate_figure": "import matplotlib.pyplot as plt\\nimport pandas as pd\\nimport io\\ndef generate_and_save_plot(df_viz_data, plot_params, plot_path_to_save):\\n  fig, ax = plt.subplots()\\n  # Example: ax.bar(df_viz_data['category_column'], df_viz_data['value_column'])\\n  ax.set_title(plot_params.get('title', 'Example Title'))\\n  ax.set_xlabel(plot_params.get('x_label', 'X Axis Example'))\\n  ax.set_ylabel(plot_params.get('y_label', 'Y Axis Example'))\\n  plt.tight_layout()\\n  plt.savefig(plot_path_to_save) # <<< ESSENTIAL LINE USING THE PROVIDED VARIABLE NAME\\n  plt.close(fig)\\n  return fig\\nfigure_object = generate_and_save_plot(df_viz_data, plot_params, plot_path_to_save) # Example of how figure_object is set",
            "data_for_visualization": {{ ... }},
            "plot_parameters": {{ ... }},
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

