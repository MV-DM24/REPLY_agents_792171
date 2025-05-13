# tools/reporter_tool.py
import json
import pandas as pd
import io
import matplotlib.pyplot as plt # Keep for type hinting and potential direct use
# import seaborn as sns # Only if your LLM is explicitly told to generate code using it AND reporter tool must support it
import streamlit as st
from crewai.tools import BaseTool 

class StreamlitReporterTool(BaseTool):
    name: str = "Streamlit Report Finalizer"
    description: str = (
        "Takes the textual analysis findings from the Data Analyst and the JSON output "
        "from the Data Visualizer (which contains Python code and data for a plot). "
        "It first attempts to parse and display any table within the analyst's findings in a user-friendly way, "
        "then displays the rest of the analyst's text. "
        "Then, it executes the visualizer's Python code to generate a plot and renders "
        "it in the Streamlit app using st.pyplot(). "
        "Returns a confirmation message or error details."
    )

    def _run(self, analyst_findings: str, visualizer_json_output: str) -> str:
        """
        Renders analyst findings and a visualization in Streamlit.

        Args:
            analyst_findings (str): The textual output from the DataAnalystAgent.
            visualizer_json_output (str): The JSON string output from the DataVisualizerAgent.
                                          This JSON should contain 'python_code_to_generate_figure',
                                          'data_for_visualization', and 'plot_parameters'.
        Returns:
            str: A message indicating success or failure of rendering.
        """
        try:
            # --- 1. Display Analyst's Findings (More User-Friendly Table Parsing) ---
            st.subheader("Analytical Insights")
            
            # Attempt to parse and display tables within ```text ... ``` blocks
            parts = analyst_findings.split("```text")
            st.markdown(parts[0]) # Display text before the first potential table block

            if len(parts) > 1:
                for i in range(1, len(parts)): # Iterate through potential table blocks
                    block_content = parts[i]
                    # Split the block to get table data and any text after it
                    table_and_after = block_content.split("```", 1)
                    table_string_data = table_and_after[0].strip()
                    text_after_current_table = table_and_after[1].strip() if len(table_and_after) > 1 else None

                    if table_string_data: # If there's content in the supposed table block
                        # st.markdown("--- DEBUG: Raw Table String Data ---")
                        # st.text(table_string_data)
                        # st.markdown("--- END DEBUG ---")
                        try:
                            lines = table_string_data.split('\n')
                            if not lines or not any(line.strip() for line in lines):
                                st.markdown("#### Data Table (Raw Text):")
                                st.text("(Table block was empty or whitespace only)")
                            else:
                                header_line_index = -1
                                # Try to find a plausible header line
                                for idx, line in enumerate(lines):
                                    # Heuristic: look for lines with multiple words, potentially common header words
                                    # This is still fragile and best replaced by analyst providing clean data for viz
                                    stripped_line = line.strip()
                                    if len(stripped_line.split()) > 2 and \
                                       ("modalita_autenticazione" in stripped_line.lower() or \
                                        "regione_residenza_domicilio" in stripped_line.lower() or \
                                        "column" in stripped_line.lower() or \
                                        "header" in stripped_line.lower() or \
                                        "category" in stripped_line.lower() or \
                                        "value" in stripped_line.lower()
                                        ): # Add more common header keywords if needed
                                        header_line_index = idx
                                        break
                                
                                if header_line_index != -1:
                                    data_to_parse_str = "\n".join(lines[header_line_index:])
                                    try:
                                        # Attempt to read with pandas, this is the tricky part for arbitrary text tables
                                        # For multi-index, it might require more sophisticated parsing or specific parameters
                                        # For now, delim_whitespace is a common attempt.
                                        df_from_text = pd.read_csv(io.StringIO(data_to_parse_str), delim_whitespace=True, on_bad_lines='skip')
                                        
                                        if not df_from_text.empty:
                                            st.markdown("#### Data Table (Attempted Parse):")
                                            st.dataframe(df_from_text)
                                        else:
                                            st.markdown("#### Data Table (Raw Text - Parsed as empty):")
                                            st.text(table_string_data)
                                    except Exception as parse_error:
                                        st.markdown("#### Data Table (Raw Text):")
                                        st.text(table_string_data) 
                                        st.caption(f"(Could not parse table for interactive display: {parse_error})")
                                else: # Header not identified robustly
                                    st.markdown("#### Data Table (Raw Text - Header not clearly identified):")
                                    st.text(table_string_data)
                        except Exception as table_processing_error:
                            st.warning(f"An error occurred while trying to process a table block: {table_processing_error}")
                            st.markdown("#### Data Table (Raw Text - Processing Error):")
                            st.text(table_string_data) # Fallback

                    if text_after_current_table:
                        st.markdown(text_after_current_table) # Display text after this block
            
            # If no ```text blocks were found, parts[0] is the whole analyst_findings
            # and the above loop for parts[1:] won't run. This is fine.

            st.markdown("---") # Separator before visualization

            # --- 2. Process and Render Visualization from Visualizer's Output ---
            st.subheader("Data Visualization")
            try:
                # Strip markdown fences from visualizer output if present, before JSON parsing
                json_to_parse_viz = visualizer_json_output
                if visualizer_json_output.startswith("```json"):
                    start_idx_viz = visualizer_json_output.find('{')
                    end_idx_viz = visualizer_json_output.rfind('}')
                    if start_idx_viz != -1 and end_idx_viz != -1 and end_idx_viz > start_idx_viz:
                        json_to_parse_viz = visualizer_json_output[start_idx_viz : end_idx_viz+1]
                    else: # Fallback
                        json_to_parse_viz = visualizer_json_output.replace("```json\n", "").replace("\n```", "").strip()
                st.markdown("--- DEBUG: String to be parsed by json.loads ---")
                st.text(json_to_parse_viz) # Display the exact string
                st.markdown(f"--- Length of string: {len(json_to_parse_viz)} ---")
                # You can also try to print the character at and around char 887
                if len(json_to_parse_viz) > 890:
                    st.text(f"Chars around 887: ...'{json_to_parse_viz[880:895]}'...")
                st.markdown("--- END DEBUG ---")
                viz_data = json.loads(json_to_parse_viz)

            except json.JSONDecodeError as e:
                error_msg = f"Error: Could not decode JSON from Visualizer: {e}. Raw Visualizer output was: {visualizer_json_output}"
                st.error(error_msg)
                return error_msg

            visualization_type = viz_data.get("visualization_type", "none")
            python_code = viz_data.get("python_code_to_generate_figure")
            data_for_viz_spec = viz_data.get("data_for_visualization")
            plot_params = viz_data.get("plot_parameters", {})
            viz_description = viz_data.get("description", "No visualization description provided.")

            if visualization_type == "none" or not python_code or not data_for_viz_spec:
                st.info(f"Visualization not generated or not applicable: {viz_data.get('description', 'No reason provided.')}")
                return f"Analyst findings displayed. Visualization was not applicable: {viz_data.get('description', 'No reason provided.')}"

            df_viz_data = None
            data_format = data_for_viz_spec.get("format", "").lower()
            data_value = data_for_viz_spec.get("value")

            if data_value is None:
                error_msg = "Error: Visualizer's 'data_for_visualization' is missing the 'value' field."
                st.error(error_msg)
                return error_msg

            try:
                if data_format == "csv_string":
                    if not isinstance(data_value, str):
                        raise ValueError("For 'csv_string' format, 'value' must be a string.")
                    df_viz_data = pd.read_csv(io.StringIO(data_value))
                elif data_format == "json_records_string":
                    if not isinstance(data_value, str):
                        raise ValueError("For 'json_records_string' format, 'value' must be a string.")
                    records = json.loads(data_value)
                    df_viz_data = pd.DataFrame(records)
                elif data_format == "json_records_list":
                     if not isinstance(data_value, list):
                        raise ValueError("For 'json_records_list' format, 'value' must be a list of dictionaries.")
                     df_viz_data = pd.DataFrame(data_value)
                elif data_format == "dict_of_lists":
                    if not isinstance(data_value, dict):
                        raise ValueError("For 'dict_of_lists' format, 'value' must be a dictionary.")
                    df_viz_data = pd.DataFrame(data_value)
                else:
                    error_msg = f"Error: Unsupported 'data_for_visualization' format from Visualizer: '{data_format}'."
                    st.error(error_msg)
                    return error_msg
            except Exception as e:
                error_msg = f"Error preparing data for visualization (Visualizer format: {data_format}): {e}. Data value was: {str(data_value)[:200]}..."
                st.error(error_msg)
                return error_msg

            local_namespace = {
                'pd': pd, 'io': io, 'plt': plt,
                'df_viz_data': df_viz_data,
                'plot_params': plot_params,
                'figure_object': None
            }
            plt.close('all')

            try:
                exec(python_code, local_namespace)
                fig = local_namespace.get('figure_object')

                if fig and isinstance(fig, plt.Figure):
                    st.markdown(f"**{plot_params.get('title', 'Visualization')}**")
                    st.markdown(f"*{viz_description}*")
                    st.pyplot(fig)
                    plt.close(fig)
                    return "Analyst findings and visualization rendered successfully in Streamlit."
                elif fig:
                    error_msg = "Error: Visualization code ran, but 'figure_object' was not a Matplotlib Figure."
                    st.warning(error_msg)
                    return error_msg
                else:
                    error_msg = "Error: Visualization code ran, but did not assign a Matplotlib Figure to 'figure_object'."
                    st.warning(error_msg)
                    return error_msg
            except Exception as e:
                error_msg = f"Error executing visualization Python code: {e}"
                st.error(error_msg)
                st.code(python_code, language="python")
                return error_msg

        except Exception as e:
            st.error(f"An unexpected error occurred in the Streamlit Reporter Tool: {e}")
            return f"Critical error in Streamlit Reporter Tool: {e}"

# To make it available for import:
reporter_tool = StreamlitReporterTool()