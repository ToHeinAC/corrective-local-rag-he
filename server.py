"""
LangGraph server configuration for all RAG agents.
Run with: uv run langgraph dev
"""
import sys
import traceback
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.server import GraphServer

# Set up logging to a file
logging.basicConfig(filename='server_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Server starting up")
logger.info(f"Python path: {sys.path}")

# Also print to stdout
print("Python path:", sys.path)

# Import all agents
try:
    print("Importing local_rag_agent...")
    from openevals_local_rag.corrective_rag_agent_local import agent as local_rag_agent, GraphState as LocalRagGraphState
    print("Successfully imported local_rag_agent")
except Exception as e:
    print(f"Error importing local_rag_agent: {e}")
    traceback.print_exc()

try:
    print("Importing web_rag_agent...")
    from openevals_local_rag.corrective_rag_agent import agent as web_rag_agent
    print("Successfully imported web_rag_agent")
except Exception as e:
    print(f"Error importing web_rag_agent: {e}")

try:
    print("Importing react_agent...")
    from openevals_local_rag.react_agent import agent as react_agent
    print("Successfully imported react_agent")
except Exception as e:
    print(f"Error importing react_agent: {e}")

try:
    print("Importing reflection_agent...")
    from openevals_local_rag.reflection_only import agent as reflection_agent
    print("Successfully imported reflection_agent")
except Exception as e:
    print(f"Error importing reflection_agent: {e}")

# Create a FastAPI app
app = FastAPI(
    title="RAG Agents Collection",
    version="0.1.0",
    description="A collection of RAG agents including a corrective RAG agent that uses local database retrieval",
)

# Define the input schema for the local RAG agent
class LocalRagAgentInput(MessagesState):
    """Input schema for the local RAG agent."""
    messages: List[Dict[str, Any]]
    database_path: Optional[str] = None
    tenant_id: Optional[str] = None

# Create a graph server
server = GraphServer(app)

# Register all agents with the server
# Make sure to use the exact same names as shown in the dropdown
# Set corrective_rag_agent_local as the first (default) agent

# The key name is what appears in the dropdown menu
try:
    logger.info("Registering corrective_rag_agent_local...")
    logger.info(f"Type of local_rag_agent: {type(local_rag_agent)}")
    print("Registering corrective_rag_agent_local...")
    print("Type of local_rag_agent:", type(local_rag_agent))
    server.register_graph("corrective_rag_agent_local", local_rag_agent, input_schema=LocalRagAgentInput, display_name="Corrective RAG Agent (Local DB)")
    logger.info("Successfully registered corrective_rag_agent_local")
    print("Successfully registered corrective_rag_agent_local")
except Exception as e:
    logger.error(f"Error registering corrective_rag_agent_local: {e}")
    logger.error(traceback.format_exc())
    print(f"Error registering corrective_rag_agent_local: {e}")
    traceback.print_exc()

try:
    print("Registering corrective_rag_agent...")
    server.register_graph("corrective_rag_agent", web_rag_agent)
    print("Successfully registered corrective_rag_agent")
except Exception as e:
    print(f"Error registering corrective_rag_agent: {e}")

try:
    print("Registering simple_react_agent...")
    server.register_graph("simple_react_agent", react_agent)
    print("Successfully registered simple_react_agent")
except Exception as e:
    print(f"Error registering simple_react_agent: {e}")

try:
    print("Registering reflection_only...")
    server.register_graph("reflection_only", reflection_agent)
    print("Successfully registered reflection_only")
except Exception as e:
    print(f"Error registering reflection_only: {e}")

# Add a health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Add an endpoint to list available databases
@app.get("/databases")
def list_databases():
    """List available databases."""
    import os
    
    # Path to the database directory
    database_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local-rag-researcher-deepseek-he",
        "database"
    )
    
    # List all directories in the database directory
    databases = []
    if os.path.exists(database_dir):
        databases = [d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d))]
    
    return {"databases": databases}

# Add an endpoint to list available tenants for a database
@app.get("/tenants/{database}")
def list_tenants(database: str):
    """List available tenants for a database."""
    import os
    
    # Path to the database directory
    database_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local-rag-researcher-deepseek-he",
        "database",
        database
    )
    
    # List all directories in the database directory
    tenants = []
    if os.path.exists(database_dir):
        tenants = [d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d))]
    
    return {"tenants": tenants}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)