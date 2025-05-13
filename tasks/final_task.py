# tasks/final_task.py
from crewai import Task
from utils.config import config
AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_final_task(query):
    return Task(
        description = f"""
**Objective:** Compile the final report for the user based on the preceding analysis of their query: '{query}'.

**Your Role:** You are the Final Results Reporter. Your sole responsibility is to take the findings provided by the Data Analyst (and Data Visualizer, when active) and synthesize them into a clear, comprehensive, and user-friendly response. You DO NOT perform new analysis or create visualizations.

**Context Provided to You (implicitly from previous tasks):**
*   **Original User Query:** '{query}'
*   **Data Analyst's Findings:** The detailed results and conclusions from the Data Analyst's work. This might include numerical results, textual explanations, or structured data.
*   **(Future) Data Visualizer's Output:** (When the Visualizer agent is active and its task is part of the context) This would include descriptions of any visualizations created, links, or data for embedding them, and any issues encountered during visualization.

**Your Steps:**

1.  **Review All Inputs:**
    *   Carefully re-read the original user query: '{query}' to ensure your response directly addresses it.
    *   Thoroughly examine the Data Analyst's findings provided to you.
    *   (When active) Review the Data Visualizer's output.

2.  **Synthesize and Structure the Final Response:**
    *   **Acknowledge the Query:** Begin by briefly referencing the user's request.
    *   **Present Analytical Results:** Clearly present the key findings, data points, and conclusions from the Data Analyst. Ensure this information is in the same language as the original query.
    *   **(Integrate Visualization Information - for future use):**
        *   If the Data Visualizer provided output and successfully created a visualization, describe the visualization (e.g., "A bar chart illustrating X shows Y..."). Do not attempt to display the image itself, but provide the description given by the Visualizer.
        *   If the Visualizer attempted but failed, or indicated that visualization was not possible/appropriate, clearly state this and explain why, based on the Visualizer's feedback (e.g., "A visualization was requested, but could not be generated because [reason from Visualizer].").
    *   **Address Limitations/Errors:** If the Data Analyst (or Visualizer) reported any difficulties, data limitations, or parts of the query that could not be answered, include a clear and concise explanation of these issues in your report.
    *   **Maintain Language:** Ensure the entire response is in the same language as the original user query.
    *   **Focus on Aggregated Data:** Remember to frame your explanations acknowledging that the analysis was based on aggregated user data from the NoiPA portal.

3.  **Produce the Final Report:**
    *   Generate the complete, polished response as your output. This is the text that will be delivered to the user.
""",
expected_output = f"""
A final, consolidated response in the language of the original query ('{query}'), addressed directly to the user.

The response MUST include:
1.  A clear presentation of the Data Analyst's findings and results, translated into the query language.
2.  (When the Data Visualizer is active and successful) A description of the visualization provided by the Data Visualizer, explaining what it represents.
3.  If visualization was requested but not possible or failed, an explanation for this, based on the Visualizer's feedback. Even in this case, the Data Analyst's results MUST still be provided.
4.  If any part of the query could not be answered by the Analyst or Visualizer, a clear explanation of why.

The response should be well-structured, easy to understand, and directly answer the user's original query to the best of the combined agents' abilities.
The entire output MUST be in the same language as the original query.
""",

    )