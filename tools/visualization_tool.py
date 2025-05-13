#tools/visualization_tool.py
from crewai.tools import BaseTool
from typing import Type, Any, Dict, List, Optional
import io
import contextlib
import json
import difflib
import os
import base64

AVAILABLE_DATA_PATHS = os.getenv("AVAILABLE_DATA_PATHS", "").split(",")  

import os
import time
import io
import contextlib
from crewai import BaseTool 


class PythonPlottingTool(BaseTool):
    name: str = "Python Plotting Tool"
    description: str = (
        "Executes Python code using Matplotlib or Seaborn to generate and save a visualization as an image file. "
        "The provided Python code MUST include a function call like `plt.savefig(plot_path_to_save)` "
        "where `plot_path_to_save` is a variable made available by this tool. "
        "This tool returns the file path to the saved plot upon successful execution, or an error message."
    )

    def _run(self, python_code: str, analyst_data_str: str = None) -> str:
        """
        Executes Python plotting code.

        Args:
            python_code (str): The Python code string to generate and save the plot.
                               This code should expect 'pd', 'np', 'plt', 'sns',
                               'analyst_data_str', and 'plot_path_to_save' to be in its scope.
            analyst_data_str (str, optional): A string representation of the analyst's data
                                              (e.g., CSV string or JSON string of a DataFrame)
                                              that the python_code can parse. Defaults to None.
        Returns:
            str: The file path to the saved plot if successful, or an error message string.
        """
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            plots_dir = "plots" 
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = int(time.time() * 1000) 
            plot_filename = f"visualization_{timestamp}.png"
            plot_path_to_save = os.path.join(plots_dir, os.path.abspath(plot_filename))

            # --- Prepare the local namespace for exec ---
            local_namespace = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'analyst_data_str': analyst_data_str,
                'plot_path_to_save': plot_path_to_save,
                'io': io
            }

            # --- Execute the LLM's Python code ---

            plt.close('all')

            exec(python_code, local_namespace)

            # --- Verify plot was saved ---
            if os.path.exists(plot_path_to_save):
                return plot_path_to_save # Success: return the path
            else:
                return (f"Error: Visualization code executed, but the plot file was not found at '{plot_path_to_save}'. "
                        f"Ensure your Python code includes `plt.savefig(plot_path_to_save)` and that `plot_path_to_save` is used correctly.")

        except Exception as e:
            # This catches Python errors within the LLM's code.
            return f"Error executing visualization code: {str(e)}"

    def _arun(self, python_code: str, analyst_data_str: str = None) -> str:
        """Async version simply calls the sync version."""
        return self._run(python_code, analyst_data_str)


visualization_tool = PythonPlottingTool()