"""
Test script to verify that we can import and use the local RAG agent.
"""
import sys
import traceback

print("Python path:", sys.path)

try:
    print("Importing local_rag_agent...")
    from openevals_local_rag.corrective_rag_agent_local import agent as local_rag_agent
    print("Successfully imported local_rag_agent")
    print("Type of local_rag_agent:", type(local_rag_agent))
except Exception as e:
    print(f"Error importing local_rag_agent: {e}")
    traceback.print_exc()

try:
    print("Importing web_rag_agent...")
    from openevals_local_rag.corrective_rag_agent import agent as web_rag_agent
    print("Successfully imported web_rag_agent")
    print("Type of web_rag_agent:", type(web_rag_agent))
except Exception as e:
    print(f"Error importing web_rag_agent: {e}")
    traceback.print_exc()
