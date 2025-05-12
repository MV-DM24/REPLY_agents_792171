#utils/tool_registry.py
from crewai.tools import BaseTool
from typing import Dict, List, Type
from tools.analysis_tool import DataAnalysisTool, analysis_tool
from tools.visualization_tool import DataVisualizationTool, visualization_tool

# Tool registry singleton
class ToolRegistry:
    _instance = None
    
    def _new_(cls):
        if cls._instance is None:
            cls.instance = super(ToolRegistry, cls).new_(cls)
            cls._instance.tools = {}
            cls._instance._initialize_tools()
        return cls._instance
    
    def _initialize_tools(self):
        """Initialize the default tools."""
        self.register_tool("DataAnalysisTool", analysis_tool)
        self.register_tool("DataVisualizationTool", visualization_tool)
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a tool in the registry."""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found in the registry.")
        return self.tools[name]
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self.tools
    
    def get_tools_by_names(self, names: List[str]) -> List[BaseTool]:
        """Get multiple tools by name."""
        return [self.get_tool(name) for name in names]

# Create a global registry instance
tool_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return tool_registry

def get_analysis_tool() -> DataAnalysisTool:
    """Get the analysis tool instance."""
    return tool_registry.get_tool("DataAnalysisTool")

def get_visualization_tool() -> DataVisualizationTool:
    """Get the visualization tool instance."""
    return tool_registry.get_tool("DataVisualizationTool")

def get_tools_for_task(task_type: str) -> List[BaseTool]:
    """
    Get the appropriate tools for a specific task type.
    
    Args:
        task_type: The type of task (e.g., 'analysis', 'visualization')
        
    Returns:
        A list of tools appropriate for the task
    """
    if task_type == 'analysis':
        return [get_analysis_tool()]
    elif task_type == 'visualization':
        return [get_visualization_tool()]
    elif task_type == 'both':
        return [get_analysis_tool(), get_visualization_tool()]
    else:
        return []