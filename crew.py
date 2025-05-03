# crew.py
import sys
import os
print("Current working directory:", os.getcwd())

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.frontman import FrontmanAgent
from agents.analyst import DataAnalystAgent
from agents.visualizer import DataVisualizerAgent
from utils.config import config 
#from utils.chroma_db import chroma_manager
from tasks.initial_task import create_initial_task
from tasks.analyst_tasks import create_analyst_task
from tasks.visualizer_tasks import create_visualization_task
from crewai import Crew, Process

frontman_agent = FrontmanAgent()
analyst_agent = DataAnalystAgent()
visualizer_agent = DataVisualizerAgent()


query = input("Enter your query: ")  
initial_task = create_initial_task(query=query)
analyst_task = create_analyst_task(query=query)
visualizer_task = create_visualization_task(query=query) 


crew = Crew(
    agents=[ analyst_agent, visualizer_agent],
    tasks=[ analyst_task, visualizer_task],
    process = Process.hierarchical,
    manager_agent = frontman_agent,
    verbose=1
)

# 4. Kickoff the Crew
crew_result = crew.kickoff()

print(f"Crew Result: {crew_result}")

