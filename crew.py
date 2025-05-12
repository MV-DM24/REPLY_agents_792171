# crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from langchain_community.chat_models import ChatLiteLLM
from utils.config import config
from utils.tool_registry import get_tools_for_task


# Define Pydantic models for structured task outputs
class AnalysisOutput(BaseModel):
    summary: str = Field(description="A summary of the analysis findings")
    insights: List[str] = Field(description="Key insights from the data analysis")
    metrics: Dict[str, float] = Field(description="Important metrics calculated during analysis")
    
class VisualizationOutput(BaseModel):
    chart_type: str = Field(description="Type of visualization created")
    visualization_data: Dict = Field(description="Data used for visualization") 
    description: str = Field(description="Description of what the visualization shows")
    plot_path: Optional[str] = Field(description="Path to the saved visualization file if applicable")

@CrewBase
class DataAnalysisCrew:
    """Data Analysis Crew for analyzing and visualizing data based on user queries."""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def _init_(self, query: str):
        """Initialize the crew with a user query."""
        self.query = query
        self.llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", api_key=config.GOOGLE_API_KEY, verbose=True)
    
    @agent
    def frontman(self) -> Agent:
        """The frontman agent coordinates the analysis and visualization."""
        return Agent(
            config=self.agents_config["frontman"],  # type: ignore[index]
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def data_analyst(self) -> Agent:
        """The data analyst agent analyzes the data."""
        return Agent(
            config=self.agents_config["data_analyst"],  # type: ignore[index]
            llm=self.llm,
            verbose=True,
            tools=get_tools_for_task('analysis')  # Use the tool registry
        )
    
    @agent
    def data_visualizer(self) -> Agent:
        """The data visualizer agent creates visualizations based on the analysis."""
        return Agent(
            config=self.agents_config["data_visualizer"],  # type: ignore[index]
            llm=self.llm,
            verbose=True,
            tools=get_tools_for_task('visualization')  # Use the tool registry
        )
    
    @task
    def initial_task(self) -> Task:
        """The initial task for the frontman agent."""
        task = Task(
            config=self.tasks_config["initial_task"],  # type: ignore[index]
        )
        # Set the agent for this task
        task.agent = self.frontman()
        return task
    
    @task
    def analysis_task(self) -> Task:
        """The analysis task for the data analyst agent."""
        task = Task(
            config=self.tasks_config["analysis_task"],  # type: ignore[index]
        )
        # Set the output_pydantic model
        task.output_pydantic = AnalysisOutput
        # Set the agent for this task
        task.agent = self.data_analyst()
        # Add tools explicitly
        task.tools = get_tools_for_task('analysis')
        return task
    
    @task
    def visualization_task(self) -> Task:
        """The visualization task for the data visualizer agent."""
        task = Task(
            config=self.tasks_config["visualization_task"],  # type: ignore[index]
        )
        # Set the output_pydantic model
        task.output_pydantic = VisualizationOutput
        # Set the agent for this task
        task.agent = self.data_visualizer()
        # Add tools explicitly
        task.tools = get_tools_for_task('visualization')
        # Add context - the visualization task depends on the analysis task
        task.context = [self.analysis_task()]
        return task
    
    @crew
    def crew(self) -> Crew:
        """Configure the crew to execute the tasks."""
        return Crew(
            agents=[
                self.frontman(),
                self.data_analyst(),
                self.data_visualizer()
            ],
            tasks=[
                self.initial_task(),
                self.analysis_task(),
                self.visualization_task()
            ],
            process=Process.sequential,
            verbose=True
        )
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatLiteLLM
from utils.config import config

# Define Pydantic models for structured task outputs
class AnalysisOutput(BaseModel):
    summary: str = Field(description="A summary of the analysis findings")
    insights: List[str] = Field(description="Key insights from the data analysis")
    metrics: Dict[str, float] = Field(description="Important metrics calculated during analysis")
    
class VisualizationOutput(BaseModel):
    chart_type: str = Field(description="Type of visualization created")
    visualization_data: Dict[str, Any] = Field(description="Data used for visualization") 
    description: str = Field(description="Description of what the visualization shows")
    plot_path: Optional[str] = Field(description="Path to the saved visualization file if applicable")

@CrewBase
class DataAnalysisCrew:
    """Data Analysis Crew for analyzing and visualizing data based on user queries."""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def _init_(self, query: str):
        """Initialize the crew with a user query."""
        self.query = query
        self.llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", api_key=config.GOOGLE_API_KEY, verbose=True)
    
    @agent
    def frontman(self) -> Agent:
        """The frontman agent coordinates the analysis and visualization."""
        return Agent(
            config=self.agents_config["frontman"],  # type: ignore[index]
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def data_analyst(self) -> Agent:
        """The data analyst agent analyzes the data."""
        return Agent(
            config=self.agents_config["data_analyst"],  # type: ignore[index]
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def data_visualizer(self) -> Agent:
        """The data visualizer agent creates visualizations based on the analysis."""
        return Agent(
            config=self.agents_config["data_visualizer"],  # type: ignore[index]
            llm=self.llm,
            verbose=True
        )
    
    @task
    def initial_task(self) -> Task:
        """The initial task for the frontman agent."""
        task = Task(
            config=self.tasks_config["initial_task"],  # type: ignore[index]
        )
        # Set the agent for this task
        task.agent = self.frontman()
        return task
    
    @task
    def analysis_task(self) -> Task:
        """The analysis task for the data analyst agent."""
        task = Task(
            config=self.tasks_config["analysis_task"],  # type: ignore[index]
        )
        # Set the output_pydantic model
        task.output_pydantic = AnalysisOutput
        # Set the agent for this task
        task.agent = self.data_analyst()
        return task
    
    @task
    def visualization_task(self) -> Task:
        """The visualization task for the data visualizer agent."""
        task = Task(
            config=self.tasks_config["visualization_task"],  # type: ignore[index]
        )
        # Set the output_pydantic model
        task.output_pydantic = VisualizationOutput
        # Set the agent for this task
        task.agent = self.data_visualizer()
        # Add context
        task.context = [self.analysis_task()]
        return task
    
    @crew
    def crew(self) -> Crew:
        """Configure the crew to execute the tasks."""
        return Crew(
            agents=[
                self.frontman(),
                self.data_analyst(),
                self.data_visualizer()
            ],
            tasks=[
                self.initial_task(),
                self.analysis_task(),
                self.visualization_task()
            ],
            process=Process.sequential,
            verbose=True
        )

