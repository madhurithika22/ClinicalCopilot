from langgraph.graph import StateGraph, END, START
from langchain_community.chat_models import ChatOllama
from .state import AgentState # Import the state you just defined

# Initialize the LLM (for all agents)
# NOTE: Update base_url if not using Docker Desktop
llm = ChatOllama(model="llama3:8b", base_url="http://host.docker.internal:11434")

def create_workflow_graph() -> StateGraph:
    """Initializes the LangGraph builder with the defined AgentState."""
    
    # Initialize the graph builder with the state dictionary
    workflow = StateGraph(AgentState)
    
    # Nodes will be added starting Day 2 by Member A and Member B
    
    return workflow

# This is the main runnable object we will compile later
agent_workflow = create_workflow_graph()

print("LangGraph builder initialized successfully.")