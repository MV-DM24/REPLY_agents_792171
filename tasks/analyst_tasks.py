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
    *   You MUST inspect columns, data types, and sample data (e.g., using `df.columns`, `df.head()`, `df.info()`) of potentially relevant files to locate the necessary information. 
    Note that relevant information might be in columns with non-obvious names or spread across multiple files.
3.  **Plan Analysis & Data Processing (Crucial for Complex Queries):**
    *   **For queries involving preferences, distributions, or correlations across multiple dimensions** (e.g., 'payment method preference by age and gender', 'access method distribution by region and age'): Your primary goal is to produce a detailed breakdown. This means calculating the **count or percentage of each sub-category for every combination of the main categories.** Do not just provide the 'most frequent' if more detail is needed for the query's intent. For example, if analyzing payment methods by age and gender, aim to produce data like: AgeGroup, Gender, PaymentMethod, Count_or_Percentage.
    * For queries asking about 'correlation':
        - If both variables involved can be made numeric, aim to provide data suitable for calculating a correlation coefficient (e.g., two columns of paired numerical data).
        - If one or more variables are categorical (e.g., 'gender', 'payment method'), 'correlation' likely means the user wants to see the *relationship or association* between them. In this case, provide a detailed contingency table (e.g., using `pd.crosstab`) or counts/percentages of one variable broken down by the other(s). This data can then be used to visualize the association (e.g., grouped/stacked bar chart, heatmap of counts). For example, for 'correlation between gender and payment method preference', provide a table with Gender, PaymentMethod, and Count/Percentage.
    *   If data needs merging from multiple files, formulate and execute a plan using appropriate keys.
    *   **Consider Visualization Needs for Data Output:** When preparing data for the `=== DATA FOR VISUALIZATION (CSV) ===` section, think about what a typical visualization for this query would require.
        *   **If the detailed, granular data is very large (e.g., many rows/categories that would make a chart unreadable):**
            *   Consider if a summarized version would be more appropriate for a *primary* visualization. For example, instead of all 50 regions, perhaps the top 10 and an 'Other' category. Or if showing many individual data points for a trend, consider binning or aggregation.
            *   **You should still perform your full detailed analysis for your textual summary,** but the data provided for visualization can be a targeted subset or aggregation if it makes the visualization clearer and more effective, and if the full detail is too overwhelming for a single chart.
            *   If you summarize/aggregate for visualization, briefly note this in your human-readable summary part.
        *   **Ensure the data is "plot-ready":** Columns should be clearly named. Data should be in a shape that's easy to plot (e.g., for bar charts: categories and values; for line charts: x-axis series and y-axis series).
4.  **Perform Analysis (Tool Use MANDATORY):** Execute the Python code to perform the planned analysis.
5.  **Formulate Response (Dual Output Required):**
    *   **Part 1: Human-Readable Summary:** Provide a clear, textual explanation of your findings in response to '{query}'. If you present small tables here (<15 rows), use Markdown. If an exact answer isn't possible but a related one is, explain the adaptation. If no answer is possible, explain why.
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
    *   A clear textual explanation answering the query, or explaining adaptations if the exact query couldn't be met, or stating impossibility if no relevant analysis could be done.
    *   If small tables are included in this summary, they should be in Markdown format.
    *   If the query asks to highlight 'significant differences', present the relevant comparative data (e.g.,     distributions, percentages). In your textual summary, describe any visually apparent or notable differences. You may also briefly mention what kind of statistical test could formally assess significance, but you are not expected to perform such tests with your current tools unless it's a simple calculation you can derive.
    *   This summary should mention any data limitations (e.g., aggregation, sample sizes).

2.  **Machine-Readable Data for Visualization (If analytical data was produced):**
    *   This section is MANDATORY if any tabular data results from your analysis.
    *   It MUST begin with the exact, single line: `=== DATA FOR VISUALIZATION (CSV) ===`
    *   Following this delimiter, provide a **single CSV formatted string** containing the detailed, granular data table relevant to the query.
        *   This CSV string MUST have headers.
        *   It should be "flat" (e.g., if your analysis involved a multi-index DataFrame, use `your_df.reset_index().to_csv(index=False)` to generate this string).
        *   **Crucially, for queries about preferences, distributions, or correlations across multiple dimensions (e.g., 'payment method by age and gender'), this CSV data MUST show the count or percentage of each sub-category for *every combination* of the main categories, not just a summary like 'most frequent'.** For instance, a table with columns like `AgeGroup, Gender, PaymentMethod, Count` or `AgeGroup, Gender, PaymentMethod, Percentage`.
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

