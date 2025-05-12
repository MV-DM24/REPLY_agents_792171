#utils/guardrail.py
import json
from typing import Tuple, Any, Dict, List, Union
from crewai import TaskOutput

def validate_json_output(result: TaskOutput) -> Tuple[bool, Union[Dict, str]]:
    """
    Validates that the task output is valid JSON with the required structure.
    
    Args:
        result: The output of the task
        
    Returns:
        A tuple containing:
        - Boolean indicating if the validation passed
        - Either the parsed JSON data or an error message
    """
    try:
        # Check if result is already parsed (for newer CrewAI versions)
        if hasattr(result, 'json_dict') and result.json_dict:
            return (True, result.json_dict)
        
        # Try to parse as JSON
        data = json.loads(result.raw if isinstance(result, TaskOutput) else result)
        return (True, data)
    except json.JSONDecodeError:
        return (False, "Output must be valid JSON. Please format your response as a proper JSON object.")

def validate_analysis_output(result: TaskOutput) -> Tuple[bool, Union[Dict, str]]:
    """
    Validates that the analysis output contains all required fields.
    
    Args:
        result: The output of the analysis task
        
    Returns:
        A tuple containing:
        - Boolean indicating if the validation passed
        - Either the parsed JSON data or an error message
    """
    # First validate it's proper JSON
    success, data = validate_json_output(result)
    if not success:
        return (False, data)
    
    # Now check for required fields
    required_fields = ['summary', 'insights', 'metrics']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return (False, f"Output is missing the following required fields: {', '.join(missing_fields)}")
    
    # Validate insights is a list
    if not isinstance(data['insights'], list):
        return (False, "The 'insights' field must be a list of strings")
    
    # Validate metrics is a dictionary
    if not isinstance(data['metrics'], dict):
        return (False, "The 'metrics' field must be a dictionary")
    
    return (True, data)

def validate_visualization_output(result: TaskOutput) -> Tuple[bool, Union[Dict, str]]:
    """
    Validates that the visualization output contains all required fields.
    
    Args:
        result: The output of the visualization task
        
    Returns:
        A tuple containing:
        - Boolean indicating if the validation passed
        - Either the parsed JSON data or an error message
    """
    # First validate it's proper JSON
    success, data = validate_json_output(result)
    if not success:
        return (False, data)
    
    # Check for required fields
    required_fields = ['chart_type', 'visualization_data', 'description']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return (False, f"Output is missing the following required fields: {', '.join(missing_fields)}")
    
    # Visualization data should be a dictionary
    if not isinstance(data['visualization_data'], dict):
        return (False, "The 'visualization_data' field must be a dictionary")
    
    return (True, data)