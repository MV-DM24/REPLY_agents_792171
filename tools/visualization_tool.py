# tools/plotting_tool.py
import os
import time
import io
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # If you want your visualizer to use it
from crewai.tools import BaseTool

class PythonPlottingTool(BaseTool):
    name: str = "Python Plotting Tool"
    description: str = (
        "Executes Python code using Matplotlib or Seaborn to generate and save a visualization as a PNG image file. "
        "The provided Python code MUST include `plt.savefig(plot_path_to_save)`. "
        "This tool provides `plot_path_to_save` and `analyst_data_str` to the code's execution scope. "
        "Returns the file path to the saved plot upon success, or an error message string."
    )

    def _run(self, python_plot_code: str, analyst_data_str: str) -> str:
        try:
            plots_dir = "plots" # Ensure this directory exists or is created by your app
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            plot_filename = f"visualization_{timestamp}.png"
            # Use absolute path for saving and returning
            plot_path_to_save = os.path.abspath(os.path.join(plots_dir, plot_filename))

            local_namespace = {
                'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'io': io,
                'analyst_data_str': analyst_data_str, # The CSV string from analyst
                'plot_path_to_save': plot_path_to_save # Path where the code should save
            }
            plt.close('all') # Clear any previous plots
            exec(python_plot_code, local_namespace)

            if os.path.exists(plot_path_to_save):
                return plot_path_to_save
            else:
                return (f"Error: Plotting code executed, but plot file not found at '{plot_path_to_save}'. "
                        "Ensure `plt.savefig(plot_path_to_save)` was called.")
        except Exception as e:
            return f"Error executing plotting code: {str(e)}"

python_plotting_tool = PythonPlottingTool() # Instance to be used by Visualizer Agent