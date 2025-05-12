#utils/yaml_loader.py
import os
import yaml
from typing import Dict, Any
from crewai import Task
from utils.guardrail import validate_analysis_output, validate_visualization_output
from utils.callbacks import track_analysis_completion, track_visualization_completion

# Dictionary of available guardrails
AVAILABLE_GUARDRAILS = {
    "validate_analysis_output": validate_analysis_output,
    "validate_visualization_output": validate_visualization_output
}

# Dictionary of available callbacks
AVAILABLE_CALLBACKS = {
    "track_analysis_completion": track_analysis_completion,
    "track_visualization_completion": track_visualization_completion
}

def load_tasks_config() -> Dict[str, Any]:
    """
    Load task configurations from YAML files.
    
    Returns:
        A dictionary containing task configurations.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tasks configuration file not found at {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_task_from_config(task_name: str, config: Dict[str, Any], inputs: Dict[str, Any] = None) -> Task:
    """
    Create a Task object from a configuration dictionary.
    
    Args:
        task_name: The name of the task to create
        config: The task configuration dictionary
        inputs: Optional inputs to format the task description
    
    Returns:
        A Task object configured according to the provided configuration
    """
    task_config = config.get(task_name)
    if not task_config:
        raise ValueError(f"Task configuration for '{task_name}' not found")
    
    # Format the description and expected output with the inputs
    description = task_config["description"]
    expected_output = task_config["expected_output"]
    
    if inputs:
        # Format the description and expected output with the inputs
        description = description.format(**inputs)
        expected_output = expected_output.format(**inputs) if expected_output else None
    
    # Get the guardrail if specified
    guardrail_name = task_config.get("guardrail")
    guardrail = AVAILABLE_GUARDRAILS.get(guardrail_name) if guardrail_name else None
    
    # Get the callback if specified
    callback_name = task_config.get("callback")
    callback = AVAILABLE_CALLBACKS.get(callback_name) if callback_name else None
    
    # Create the Task object
    task = Task(
        name=task_name,
        description=description,
        expected_output=expected_output
    )
    
    # Set additional Task attributes if specified in the configuration
    if guardrail:
        task.guardrail = guardrail
    
    if callback:
        task.callback = callback
    
    if "max_retries" in task_config:
        task.max_retries = task_config["max_retries"]
    
    if "output_file" in task_config:
        task.output_file = task_config["output_file"]
    
    if "create_directory" in task_config:
        task.create_directory = task_config["create_directory"]
    
    # Return the configured Task object
    return task