from utils.config import config
from crewai.tools import BaseTool
from typing import Type, Any, Dict, List, Optional
import io
import contextlib
import json
import difflib

class DataVisualizationTool(BaseTool):
    name: str = "Python Visualization Executor"
    description: str = "Create data visualizations using matplotlib, seaborn, and plotly based on pandas dataframes"

    def _run(self, code: str) -> str:
        """Execute Python code for data visualization and return interactive visualizations."""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import json
            import difflib
            import io
            import contextlib
            from utils.config import config # Import config

            # Add column matching function
            def match_column_name(df, requested_col):
                """Find the closest matching column name in the dataframe."""
                if requested_col in df.columns:
                    return requested_col

                # Find closest match if exact match not found
                matches = difflib.get_close_matches(requested_col, df.columns, n=1, cutoff=0.6)
                if matches:
                    return matches[0]
                return requested_col  # Return original if no match found

            # Define helper visualization functions for plotly (interactive)
            def create_interactive_bar(df, x_col, y_col, title=None, color=None):
                """Create an interactive bar chart using plotly"""
                # Match column names to best available in dataset
                x_col_matched = match_column_name(df, x_col)
                y_col_matched = match_column_name(df, y_col)
                color_matched = match_column_name(df, color) if color else None

                fig = px.bar(df, x=x_col_matched, y=y_col_matched, color=color_matched,
                             title=title or f"Bar Chart of {y_col_matched} by {x_col_matched}")
                return fig

            def create_interactive_pie(df, names_col, values_col, title=None):
                """Create an interactive pie chart using plotly"""
                # Match column names
                names_col_matched = match_column_name(df, names_col)
                values_col_matched = match_column_name(df, values_col)

                fig = px.pie(df, names=names_col_matched, values=values_col_matched,
                             title=title or f"Distribution of {values_col_matched} by {names_col_matched}")
                return fig

            def create_interactive_line(df, x_col, y_col, title=None, color=None):
                """Create an interactive line chart using plotly"""
                # Match column names
                x_col_matched = match_column_name(df, x_col)
                y_col_matched = match_column_name(df, y_col)
                color_matched = match_column_name(df, color) if color else None

                fig = px.line(df, x=x_col_matched, y=y_col_matched, color=color_matched,
                              title=title or f"Trend of {y_col_matched} over {x_col_matched}")
                return fig

            def create_interactive_scatter(df, x_col, y_col, title=None, color=None, size=None):
                """Create an interactive scatter plot using plotly"""
                # Match column names
                x_col_matched = match_column_name(df, x_col)
                y_col_matched = match_column_name(df, y_col)
                color_matched = match_column_name(df, color) if color else None
                size_matched = match_column_name(df, size) if size else None

                fig = px.scatter(df, x=x_col_matched, y=y_col_matched, color=color_matched, size=size_matched,
                                title=title or f"Relationship between {x_col_matched} and {y_col_matched}")
                return fig

            def create_interactive_heatmap(df_corr, title=None):
                """Create an interactive heatmap for correlation matrix using plotly"""
                fig = px.imshow(df_corr, text_auto=True, aspect="auto",
                               title=title or "Correlation Heatmap")
                return fig

            def create_interactive_histogram(df, column, title=None, nbins=None, color=None):
                """Create an interactive histogram using plotly"""
                # Match column names
                column_matched = match_column_name(df, column)
                color_matched = match_column_name(df, color) if color else None

                fig = px.histogram(df, x=column_matched, nbins=nbins, color=color_matched,
                                  title=title or f"Distribution of {column_matched}")
                return fig

            def create_interactive_box(df, x_col=None, y_col=None, title=None, color=None):
                """Create an interactive box plot using plotly"""
                # Match column names
                x_col_matched = match_column_name(df, x_col) if x_col else None
                y_col_matched = match_column_name(df, y_col) if y_col else None
                color_matched = match_column_name(df, color) if color else None

                fig = px.box(df, x=x_col_matched, y=y_col_matched, color=color_matched,
                            title=title or f"Box Plot of {y_col_matched or x_col_matched}")
                return fig

            def plotly_to_json(fig):
                """Convert plotly figure to JSON for interactive display"""
                return json.dumps(fig.to_dict())

            # Create local namespace with our imports and helper functions
            local_namespace = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'px': px,
                'go': go,
                'make_subplots': make_subplots,
                'AVAILABLE_DATA_PATHS': config.AVAILABLE_DATA_PATHS, #Access from config
                # Add helper visualization functions
                'create_interactive_bar': create_interactive_bar,
                'create_interactive_pie': create_interactive_pie,
                'create_interactive_line': create_interactive_line,
                'create_interactive_scatter': create_interactive_scatter,
                'create_interactive_heatmap': create_interactive_heatmap,
                'create_interactive_histogram': create_interactive_histogram,
                'create_interactive_box': create_interactive_box,
                'plotly_to_json': plotly_to_json,
                'match_column_name': match_column_name  # Make column matching available to user code
            }

            # Execute the code
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                exec(code, local_namespace)
            output = output_buffer.getvalue()

            # If there's a return_value variable in the code's namespace, return it
            if 'return_value' in local_namespace:
                result = local_namespace['return_value']
                return output + "\n" + str(result)
            else:
                return output or "Code executed successfully, but no output was produced."

        except Exception as e:
            return f"Error executing visualization code: {str(e)}"

    def _arun(self, code: str) -> str:
        """Async version simply calls the sync version."""
        return self._run(code)

visualization_tool = DataVisualizationTool()