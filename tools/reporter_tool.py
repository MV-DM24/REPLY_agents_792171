# tools/reporter_tool.py
import json
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import streamlit as st
from crewai.tools import BaseTool

class StreamlitReporterTool(BaseTool):
    name: str = "Streamlit Report Finalizer"
    description: str = (
        "Takes textual analysis findings from the Data Analyst and a JSON output from the "
        "Data Visualizer. The Visualizer's JSON contains Python code defining a plotting function, "
        "the data for the plot, and plot parameters. This tool displays the analyst's findings, "
        "then executes the plotting function with the provided data and parameters, "
        "and renders the resulting Matplotlib Figure in Streamlit using st.pyplot(). "
        "Returns a confirmation message or error details."
    )

    def _run(self, analyst_findings: str, visualizer_json_output: str) -> str:
        """
        Renders analyst findings and a visualization in Streamlit.
        """
        try:
            # --- 1. Display Analyst's Findings (with improved table parsing) ---
            st.subheader("Analytical Insights")
            parts = analyst_findings.split("```text")
            st.markdown(parts[0])
            if len(parts) > 1:
                for i in range(1, len(parts)):
                    block_content = parts[i]
                    table_and_after = block_content.split("```", 1)
                    table_string_data = table_and_after[0].strip()
                    text_after_current_table = table_and_after[1].strip() if len(table_and_after) > 1 else None
                    if table_string_data:
                        try:
                            lines = table_string_data.split('\n')
                            if not lines or not any(line.strip() for line in lines):
                                st.markdown("##### Data Table (Raw Text - Block was empty):")
                                st.text("(Table block was empty or whitespace only)")
                            else:
                                # Basic heuristic for header, can be improved
                                header_line_index = 0
                                for idx, line_content in enumerate(lines):
                                    if len(line_content.strip().split()) > 1: # Simple check for multiple words
                                        header_line_index = idx
                                        break
                                data_to_parse_str = "\n".join(lines[header_line_index:])
                                try:
                                    df_from_text = pd.read_csv(io.StringIO(data_to_parse_str), delim_whitespace=True, on_bad_lines='skip')
                                    if not df_from_text.empty:
                                        st.markdown("##### Data Table (Attempted Parse):")
                                        st.dataframe(df_from_text)
                                    else:
                                        st.markdown("##### Data Table (Raw Text - Parsed as empty):")
                                        st.text(table_string_data)
                                except Exception: # Broad exception for parsing issues
                                    st.markdown("##### Data Table (Raw Text - Could not parse nicely):")
                                    st.text(table_string_data)
                        except Exception as e_table:
                            st.warning(f"Could not process a text table from analyst: {e_table}")
                            st.text(table_string_data) # Show raw if any error
                    if text_after_current_table:
                        st.markdown(text_after_current_table)
            st.markdown("---")

            # --- 2. Process and Render Visualization ---
            st.subheader("Data Visualization")
            st.subheader("Data Visualization")
            try:
                json_to_parse_viz = visualizer_json_output
                if visualizer_json_output.startswith("```json"): # Strip markdown
                    start_idx_viz = visualizer_json_output.find('{')
                    end_idx_viz = visualizer_json_output.rfind('}')
                    if start_idx_viz != -1 and end_idx_viz != -1 and end_idx_viz > start_idx_viz:
                        json_to_parse_viz = visualizer_json_output[start_idx_viz : end_idx_viz+1]
                    else:
                        json_to_parse_viz = visualizer_json_output.replace("```json\n", "").replace("\n```", "").strip()
                viz_data = json.loads(json_to_parse_viz)
            except json.JSONDecodeError as e:
                error_msg = f"Error: Could not decode JSON from Visualizer: {e}. Raw: {visualizer_json_output[:500]}..."
                st.error(error_msg)
                return error_msg

            visualization_type = viz_data.get("visualization_type", "none")
            plot_path = viz_data.get("plot_path")
            plot_params = viz_data.get("plot_parameters", {})
            viz_description = viz_data.get("description", "No visualization description provided.")

            if visualization_type != "none" and plot_path:
                if os.path.exists(plot_path):
                    st.markdown(f"**{plot_params.get('title', 'Visualization')}**")
                    st.markdown(f"*{viz_description}*")
                    st.image(plot_path) # Display the image from the file path
                    return "Analyst findings and visualization image rendered successfully in Streamlit."
                else:
                    st.error(f"Visualization Error: Plot image file not found at path: {plot_path}. The Visualizer may have failed to save it.")
                    st.info(f"Visualizer's description: {viz_description}") # Show why it thought it succeeded or failed
                    return f"Analyst findings displayed. Visualization image not found at {plot_path}."
            else:
                st.info(f"Visualization not generated or not applicable: {viz_description}")
                return f"Analyst findings displayed. Visualization not applicable or path missing: {viz_description}"

        except Exception as e_main:
            st.error(f"An critical error occurred in the Streamlit Reporter Tool: {e_main}")
            return f"Critical error in Streamlit Reporter Tool: {e_main}"
           

# To make it available for import:
reporter_tool = StreamlitReporterTool()