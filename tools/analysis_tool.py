#tools/analysis_tool.py

import os
from crewai.tools import BaseTool
from typing import Type, Any, Dict, List, Optional
import io
import contextlib
AVAILABLE_DATA_PATHS = os.environ.get("AVAILABLE_DATA_PATHS", "").split(",")

class DataAnalysisTool(BaseTool):
    name: str = "Python Pandas Code Executor"
    description: str = "Execute Python code for data analysis using pandas. Code must use file paths from AVAILABLE_DATA_PATHS."

    def _run(self, code: str) -> str:
        """Execute Python code for data analysis and return the results."""
        try:
            import pandas as pd
            import numpy as np
            import io
            import contextlib
            from utils.config import config 

            local_namespace = {
                'pd': pd,
                'np': np,
                'AVAILABLE_DATA_PATHS': config.AVAILABLE_DATA_PATHS 
            }

            output_buffer = io.StringIO()

            with contextlib.redirect_stdout(output_buffer):
                exec(code, local_namespace)

            output = output_buffer.getvalue()

            if 'return_value' in local_namespace:
                result = local_namespace['return_value']
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    return output + "\n" + str(result)
                else:
                    return output + "\n" + str(result)
            else:
                return output or "Code executed successfully, but no output was produced."

        except Exception as e:
            return f"Error executing code: {str(e)}"

    def _arun(self, code: str) -> str:
        """Async version simply calls the sync version."""
        return self._run(code)

analysis_tool = DataAnalysisTool()