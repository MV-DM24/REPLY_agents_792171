#utils/callbacks.py


import logging
from crewai import TaskOutput

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("crew_tasks.log"), logging.StreamHandler()]
)

logger = logging.getLogger("crew_tasks")

def log_task_completion(output: TaskOutput):
    """
    Callback function to log task completion.
    
    Args:
        output: The TaskOutput object containing the task results
    """
    task_summary = output.summary if hasattr(output, 'summary') else output.description[:50] + "..."
    logger.info(f"Task completed: {task_summary}")
    
    # Log output format
    if hasattr(output, 'pydantic') and output.pydantic:
        logger.info(f"Output format: Pydantic model ({type(output.pydantic)._name_})")
    elif hasattr(output, 'json_dict') and output.json_dict:
        logger.info(f"Output format: JSON")
    else:
        logger.info("Output format: Raw text")
    
    # Log length of output
    if hasattr(output, 'raw'):
        output_length = len(output.raw) if output.raw else 0
        logger.info(f"Output length: {output_length} characters")

def track_analysis_completion(output: TaskOutput):
    """
    Callback function specifically for the analysis task.
    
    Args:
        output: The TaskOutput object containing the analysis results
    """
    log_task_completion(output)
    
    if hasattr(output, 'pydantic') and output.pydantic:
        # Log number of insights
        num_insights = len(getattr(output.pydantic, 'insights', []))
        logger.info(f"Analysis produced {num_insights} insights")
        
        # Log metrics
        metrics = getattr(output.pydantic, 'metrics', {})
        logger.info(f"Analysis produced {len(metrics)} metrics")

def track_visualization_completion(output: TaskOutput):
    """
    Callback function specifically for the visualization task.
    
    Args:
        output: The TaskOutput object containing the visualization results
    """
    log_task_completion(output)
    
    if hasattr(output, 'pydantic') and output.pydantic:
        # Log visualization type
        chart_type = getattr(output.pydantic, 'chart_type', 'unknown')
        logger.info(f"Created visualization of type: {chart_type}")
        
        # Log file path if available
        plot_path = getattr(output.pydantic, 'plot_path', None)
        if plot_path:
            logger.info(f"Visualization saved to: {plot_path}")