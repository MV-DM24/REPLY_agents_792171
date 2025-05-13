# tasks/final_task.py
from crewai import Task
from utils.config import config
AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_final_reporting_task(
    reporter_agent, 
    original_user_query: str,
    analyst_findings_context_name: str, 
    visualizer_json_context_name: str   
):
    """
    Creates the final task for the ReporterAgent to use its Streamlit tool.

    Args:
        reporter_agent: The instance of ReporterAgent.
        original_user_query (str): The original query from the user.
        analyst_findings_context_name (str): Placeholder for the analyst's textual findings.
        visualizer_json_context_name (str): Placeholder for the visualizer's JSON output.

    Returns:
        Task: An instance of a CrewAI Task.
    """
    description_for_reporter_task = f"""
**Objective:** Your final mission is to present the complete findings for the user's query: '{original_user_query}'. You will use your 'Streamlit Report Finalizer' tool to achieve this.

**Your Role:** You are the Chief Communications Officer & Streamlit Report Publisher. You will take the textual analytical findings from the Data Analyst and the JSON blueprint (Python code + data) from the Data Visualizer, and pass them to your specialized tool for rendering in Streamlit.

**Inputs Provided to You (from preceding tasks' context):**
*   **Original User Query:** '{original_user_query}' (for overall context, though the tool primarily needs the direct outputs below).
*   **Data Analyst's Findings:** (Available as '{analyst_findings_context_name}') The detailed textual results and conclusions from the Data Analyst.
*   **Data Visualizer's JSON Output:** (Available as '{visualizer_json_context_name}') The JSON string from the Visualizer containing `python_code_to_generate_figure`, `data_for_visualization`, `plot_parameters`, and `description`.

**Your Steps:**

1.  **Gather Inputs:**
    *   Confirm you have access to the Data Analyst's findings (as a string via '{analyst_findings_context_name}').
    *   Confirm you have access to the Data Visualizer's complete JSON output (as a string via '{visualizer_json_context_name}').

2.  **Invoke the 'Streamlit Report Finalizer' Tool:**
    *   You MUST use the 'Streamlit Report Finalizer' tool.
    *   Provide the tool with two arguments:
        1.  `analyst_findings`: The string output from the Data Analyst.
        2.  `visualizer_json_output`: The JSON string output from the Data Visualizer.
    *   The tool will handle parsing the visualizer's JSON, executing its Python code to generate a plot, and rendering both the analyst's text and the plot directly in the Streamlit application.

3.  **Formulate Final Textual Summary:**
    *   The 'Streamlit Report Finalizer' tool will return a message (e.g., "Analyst findings and visualization rendered successfully in Streamlit." or an error message).
    *   Your final output for this task MUST be a concise textual summary based on this message from the tool.
    *   **Do NOT re-describe the analysis or visualization details in your output.** The tool handles the direct display in Streamlit. Your output is purely a confirmation of the rendering action.
    *   If the tool reports an error, include that error message in your summary.

**Example Output Scenarios from You (the Agent):**
*   If tool succeeds: "The requested analysis and visualization have been rendered in the Streamlit application. The Streamlit Report Finalizer tool confirmed: [Tool's success message here]."
*   If tool has issues: "The analytical findings were processed. However, the Streamlit Report Finalizer tool encountered an issue: [Tool's error message here]."
"""

    expected_output_for_reporter_task = f"""
A concise textual summary confirming the outcome of using the 'Streamlit Report Finalizer' tool.

This summary MUST:
1.  Acknowledge that the 'Streamlit Report Finalizer' tool was used.
2.  Relay the success or failure message returned by the tool.
3.  Be brief and focused on the rendering action, not a re-statement of the analytical or visual content (as that content is displayed directly by the tool in Streamlit).
4.  Be in the same language as the original user query ('{original_user_query}').

Example of a good output if successful:
"The final report, including analysis and visualizations, has been prepared and displayed in the application using the Streamlit Report Finalizer. Tool status: Analyst findings and visualization rendered successfully in Streamlit."

Example of a good output if there was an issue with visualization rendering:
"The analytical findings have been prepared for display. The Streamlit Report Finalizer tool reported an issue with the visualization: Error executing visualization Python code: [specific error]. The analyst's findings should still be visible."
"""

    return Task(
        description=description_for_reporter_task,
        expected_output=expected_output_for_reporter_task,
        agent=reporter_agent
    )