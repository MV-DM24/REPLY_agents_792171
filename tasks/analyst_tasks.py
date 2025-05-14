# tasks/analyst_tasks.py
from crewai import Task
from utils.config import config
from tools.analysis_tool import analysis_tool

AVAILABLE_DATA_PATHS = config.AVAILABLE_DATA_PATHS

def create_analyst_task(analyst_agent, query: str):
    description_for_analyst_task = f"""
Analyze the data based on the provided query: '{query}'.
You have access to data files via the `AVAILABLE_DATA_PATHS` variable in your Python code execution environment.

**STEPS TO FOLLOW:**

1.  **Understand Query:** Determine the specific information and level of detail required by the query: '{query}'.

2.  **Explore Data (Tool Use MANDATORY):**
    *   Systematically use the 'Python Code Executor' tool to explore ALL data files available.
    *   You MUST inspect columns, data types, and sample data (e.g., using `df.columns`, `df.head()`, `df.info()`) 
    of potentially relevant files to locate the necessary information. 
    Note that relevant information might be in columns with non-obvious names or spread across multiple files.

3.  **Devise Your Analytical Plan (MANDATORY for ALL queries, especially complex ones):**
    *   For the current query: '{query}', you MUST apply your structured thinking process 
        (Deconstruct, Identify Data, Formulate Step-by-Step Plan, Self-Correct) as outlined in your core 
        operational guidelines [referring to the agent's goal].
    *   **Specifically for this query, consider:**
        *   What are the key entities, properties, and relationships mentioned in '{query}'?
        *   Which datasets and specific columns (remembering Italian names and the 'numero' column for counts) 
            are essential?
        *   What sequence of operations (filtering, merging on appropriate keys, grouping by relevant dimensions, 
            calculating sums of 'numero' for counts/percentages, etc.) will directly answer this query?
    *   **Guidance for Common Query Types (apply if relevant to '{query}'):**
        *   **Distributions/Preferences (e.g., 'payment method by age'):** Ensure your plan calculates the count 
            (sum of `numero`) or percentage for *each sub-category* across all relevant dimension combinations to 
            provide a detailed breakdown (e.g., using `groupby().agg({{'numero': 'sum'}}).unstack()`, 
            `pd.crosstab(..., values=df['numero'], aggfunc='sum')`). Do not just state the 'most frequent' 
            if more detail is implied for a distribution.
        *   **Correlations/Associations:** If the query asks for a 'correlation' and involves categorical data, 
            plan to show the *relationship* by providing a detailed contingency table or counts/percentages of one 
            variable broken down by others.
    *   **Consider Visualization Needs for Data Output:** When preparing data for the `=== DATA FOR VISUALIZATION (CSV) ===` section, think about what a typical visualization for this query would require.
        *   **If the detailed, granular data is very large (e.g., many rows/categories that would make a chart 
            unreadable):**
            *   Consider if a summarized version would be more appropriate for a *primary* visualization. 
            For example, instead of all 50 regions, perhaps the top 10 and an 'Other' category. Or if showing many 
            individual data points for a trend, consider binning or aggregation.
            *   **You should still perform your full detailed analysis for your textual summary,** but the data 
                provided for visualization can be a targeted subset or aggregation if it makes the visualization 
                clearer and more effective, and if the full detail is too overwhelming for a single chart.
            *   If you summarize/aggregate for visualization, briefly note this in your human-readable summary part.
        *   **Ensure the data is "plot-ready":** Columns should be clearly named. Data should be in a shape that's 
            easy to plot (e.g., for bar charts: categories and values; for line charts: x-axis series and y-axis 
            series).

4.  **Perform Analysis (Tool Use MANDATORY):** Execute the Python code to perform the planned analysis.

5.  **Formulate Response (Dual Output Required):**
    *   **Part 1: Human-Readable Summary:** Provide a clear, textual explanation of your findings in response to '{query}'. If you present small tables here (<15 rows), use Markdown.
        If tables are large and you state "see the table below X", always provide a concise textual summary of its main points INSTEAD 
        of referring to a non-existent table. 
        If an exact answer isn't possible but a related one is, explain the adaptation. If no answer is possible, explain why.
    *   **Part 2: Machine-Readable Data for Visualization:** After your textual summary, include a clearly delimited section starting with the exact line:
        `=== DATA FOR VISUALIZATION (CSV) ===`
        Below this line, provide the detailed, granular data table (produced in Step 3 for complex queries, or simpler data for others) as a single CSV string. This CSV should be flat (e.g., use `df.reset_index().to_csv(index=False)` if starting from a multi-index DataFrame) and directly parsable. This part is CRUCIAL for any downstream visualization tasks. This should be the data (potentially summarized or targeted as decided in Step 3) intended for direct use by the Visualizer.


Think step-by-step and ensure your Python code for the 'Python Code Executor' tool achieves these objectives.
Your final output string must contain both Part 1 and Part 2 if data is produced.
"""

    expected_output_for_analyst_task = f"""
A comprehensive response string that addresses the query '{query}' and is structured for both human reading and downstream machine processing. The response MUST be in the same language as the query.

The response string MUST contain:

1.  **Human-Readable Summary:**
    *   Begin with a clear textual explanation answering the core of the query.
    *   **Presenting Tabular Data in Summary:**
        *   If your analysis produces tabular data that directly presents the key findings, you MUST include a representation of this data within this summary.
        *   **For small tables (e.g., less than 15 rows AND less than 10 columns):** Format the entire table as Markdown directly within this summary.
        *   **For larger tables:**
            1.  Provide a concise textual summary of the table's main purpose and overall trends.
            2.  Present a **truncated version of the table in Markdown format**, showing, for example, the **first 5-10 rows and the last 2-3 rows**, or a relevant subset if the query implies a focus. Clearly indicate that the table is truncated (e.g., by adding a note like "... (and X more rows not shown) ..." at the end of the Markdown table).
            3.  Mention that the full, detailed data is available in the `=== DATA FOR VISUALIZATION (CSV) ===` section for complete review and will be used for any generated visualizations.
    *   After presenting any direct findings or tables (full or truncated), provide a brief interpretation or highlight key insights from the data (e.g., "SPID is the most used method...", "Region X shows the highest Y...").
    *   If you adapted the analysis because the exact query couldn't be met, clearly explain the adaptation here.
    *   State any necessary caveats or data limitations (e.g., data aggregation).
    *   This summary part should be self-contained and understandable on its own.

2.  **Machine-Readable Data for Visualization (If analytical data was produced):**
    *   This section is MANDATORY if any tabular data results from your analysis.
    *   It MUST begin with the exact, single line: `=== DATA FOR VISUALIZATION (CSV) ===`
    *   Following this delimiter, provide a **single CSV formatted string** containing the detailed, granular data 
        table relevant to the query.
        *   This CSV string MUST have headers.
        *   It should be "flat" (e.g., if your analysis involved a multi-index DataFrame, 
            use `your_df.reset_index().to_csv(index=False)` to generate this string).
        *   **Crucially, for queries about preferences, distributions, or correlations across multiple dimensions 
            (e.g., 'payment method by age and gender'), this CSV data MUST show the count or percentage of each 
            sub-category for *every combination* of the main categories, not just a summary like 'most frequent'.** 
            For instance, a table with columns like `AgeGroup, Gender, PaymentMethod, Count` or 
            `AgeGroup, Gender, PaymentMethod, Percentage`.
    *   **Data Suitability for Visualization:**
        *   The data in this CSV string should be appropriately shaped and, if necessary, summarized or subsetted from your full analysis to be suitable for creating a clear and effective single visualization.
        *   For example, if analyzing data across many regions, you might provide data for the top N regions plus an "Other" category for the visualization CSV, even if your textual summary discusses all regions.
        *   The goal is to provide the Visualizer with data that is directly plottable and won't result in an overly cluttered or unreadable chart due to excessive categories or data points, unless the query specifically demands such detail in the plot.
    *   **Granularity for Complex Queries:** Even if summarized for visualization, for queries about preferences, distributions, or correlations, the underlying *logic* of your analysis should still aim to calculate the detailed breakdowns (count/percentage of each sub-category for every combination) before any summarization for the visualization CSV occurs. The summary should be a derivative of this detailed analysis.

If the query cannot be answered at all (even with an adapted analysis), the response should only contain the Human-Readable Summary explaining why, and the "=== DATA FOR VISUALIZATION (CSV) ===" section should be omitted or explicitly state "No data produced for visualization due to [reason]".
"""
    return Task(
        description=description_for_analyst_task,
        expected_output=expected_output_for_analyst_task,
        agent=analyst_agent
    )

