# tools/reporter_tool.py
import json
import pandas as pd
import io
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
            python_code_str = viz_data.get("python_code_to_generate_figure")
            data_for_viz_spec = viz_data.get("data_for_visualization")
            plot_params = viz_data.get("plot_parameters", {}) # Expected to contain title, x_label, y_label
            viz_description = viz_data.get("description", "No visualization description provided.")

            if visualization_type == "none" or not python_code_str or not data_for_viz_spec:
                st.info(f"Visualization not generated or not applicable: {viz_data.get('description', 'No reason provided.')}")
                return f"Analyst findings displayed. Visualization not applicable: {viz_data.get('description', 'No reason provided.')}"

            # Prepare DataFrame for the plotting function
            df_viz_data = None
            data_format = data_for_viz_spec.get("format", "").lower()
            data_value = data_for_viz_spec.get("value")

            if data_value is None:
                error_msg = "Error: Visualizer's 'data_for_visualization' is missing 'value'."
                st.error(error_msg)
                return error_msg
            try:
                if data_format == "csv_string":
                    df_viz_data = pd.read_csv(io.StringIO(data_value))
                elif data_format == "json_records_string":
                    records = json.loads(data_value)
                    df_viz_data = pd.DataFrame(records)
                elif data_format == "json_records_list":
                    df_viz_data = pd.DataFrame(data_value)
                elif data_format == "dict_of_lists":
                    df_viz_data = pd.DataFrame(data_value)
                else:
                    error_msg = f"Error: Unsupported 'data_for_visualization.format': '{data_format}'."
                    st.error(error_msg)
                    return error_msg
                if df_viz_data.empty and data_value: # Data value was there but resulted in empty df
                    st.warning(f"Warning: Data for visualization (format: {data_format}) resulted in an empty DataFrame.")
            except Exception as e:
                error_msg = f"Error preparing DataFrame for viz (format: {data_format}): {e}. Value: {str(data_value)[:200]}..."
                st.error(error_msg)
                return error_msg

            # Execute the Python code from the visualizer to get the plotting function
            # and then call it.
            local_namespace = {'pd': pd, 'io': io, 'plt': plt} # Basic imports for the exec scope
            figure_object = None
            plotting_function_name = "generate_visualization_figure" # CONVENTION

            try:
                # The visualizer's code should define a function, e.g., generate_visualization_figure
                # And that function should accept df_data_for_plot and plot parameters.
                code_to_exec = python_code_str
                if f"def {plotting_function_name}(" not in python_code_str:
                    # If the LLM didn't define the function, try to wrap its code
                    # This is a fallback and ideally the LLM follows the convention.
                    code_to_exec = (
                        #import matplotlib.pyplot as plt
                        #import pandas as pd
                        #import io
                        f"def {plotting_function_name}(df_data_for_plot, title_param, xlabel_param, ylabel_param):\n"
                        f"    # Intend for LLM's code to be effectively placed here\n"
                        f"    # This is a simple wrapper; complex LLM code might not fit perfectly.\n"
                        f"    # The LLM should ideally provide the full function definition.\n"
                        f"    fig, ax = plt.subplots()\n" # Default fig if LLM code doesn't make one
                        f"    # Try to execute LLM's core plotting logic within this structure\n"
                        f"    # This part is tricky if LLM code is not a self-contained function body."
                    )
                
                exec(python_code_str, local_namespace) # Defines the function in local_namespace

                if plotting_function_name in local_namespace and callable(local_namespace[plotting_function_name]):
                    plot_func = local_namespace[plotting_function_name]
                    # Call the function with the prepared DataFrame and plot parameters
                    figure_object = plot_func(
                        df_viz_data, # The prepared DataFrame
                        plot_params.get("title", "Generated Visualization"),
                        plot_params.get("x_label", "X-Axis"),
                        plot_params.get("y_label", "Y-Axis")
                        # Pass other params if your function expects them (e.g., colors, chart_type hints)
                    )
                else:
                    raise NameError(f"Plotting function '{plotting_function_name}' not found or not callable in Visualizer's code.")

                if figure_object and isinstance(figure_object, plt.Figure):
                    st.markdown(f"**{plot_params.get('title', 'Visualization')}**")
                    st.markdown(f"*{viz_description}*")
                    st.pyplot(figure_object)
                    plt.close(figure_object) # Good practice
                    return "Analyst findings and visualization rendered successfully in Streamlit."
                elif figure_object:
                    error_msg = "Error: Plotting function executed, but did not return a Matplotlib Figure."
                    st.warning(error_msg)
                    return error_msg
                else: # Should be caught by NameError or if function returned None
                    error_msg = "Error: Plotting function did not produce a figure object."
                    st.warning(error_msg)
                    return error_msg

            except Exception as e_exec:
                error_msg = f"Error executing visualization Python code: {e_exec}"
                st.error(error_msg)
                st.markdown("--- DEBUG: Failed Visualization Code ---")
                st.code(python_code_str, language="python")
                if df_viz_data is not None:
                    st.markdown("#### DataFrame Passed to Code (`df_viz_data`):")
                    st.dataframe(df_viz_data.head())
                    buffer = io.StringIO()
                    df_viz_data.info(buf=buffer)
                    st.text(f"DataFrame Info:\n{buffer.getvalue()}")
                else:
                    st.text("df_viz_data was None or not prepared.")
                st.markdown("--- END DEBUG ---")
                return error_msg

        except Exception as e_main:
            st.error(f"An critical error occurred in the Streamlit Reporter Tool: {e_main}")
            return f"Critical error in Streamlit Reporter Tool: {e_main}"

# To make it available for import:
reporter_tool = StreamlitReporterTool()