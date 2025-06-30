import json
import asyncio
import os
import sys
from datetime import datetime

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool

from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_HELPFULNESS_PROMPT,
)

# Use the absolute path to local-rag-researcher-deepseek-he directory
local_rag_path = '/home/he/ai/dev/langgraph/local-rag-researcher-deepseek-he'

# Import local RAG functionality using absolute imports
sys.path.insert(0, local_rag_path)

# For LangGraph server compatibility, we use try/except to handle imports
try:
    # Try direct imports first
    from src.assistant.v1_1.vector_db_v1_1 import search_documents, get_embedding_model_path
    from src.assistant.v1_1.rag_helpers_v1_1 import format_documents_as_plain_text
    from src.assistant.v1_1.configuration_v1_1 import Configuration, get_config_instance
    print("Direct imports successful")
except ImportError:
    # If direct imports fail, try absolute imports
    module_path = os.path.join(local_rag_path, 'src', 'assistant', 'v1_1')
    print(f"Trying absolute imports from {module_path}")
    sys.path.insert(0, module_path)
    
    try:
        import importlib.util
        
        # Import vector_db_v1_1
        vector_db_spec = importlib.util.spec_from_file_location(
            "vector_db_v1_1", os.path.join(module_path, "vector_db_v1_1.py"))
        vector_db = importlib.util.module_from_spec(vector_db_spec)
        vector_db_spec.loader.exec_module(vector_db)
        search_documents = vector_db.search_documents
        get_embedding_model_path = vector_db.get_embedding_model_path
        
        # Import rag_helpers_v1_1
        rag_helpers_spec = importlib.util.spec_from_file_location(
            "rag_helpers_v1_1", os.path.join(module_path, "rag_helpers_v1_1.py"))
        rag_helpers = importlib.util.module_from_spec(rag_helpers_spec)
        rag_helpers_spec.loader.exec_module(rag_helpers)
        format_documents_as_plain_text = rag_helpers.format_documents_as_plain_text
        
        # Import configuration_v1_1
        config_spec = importlib.util.spec_from_file_location(
            "configuration_v1_1", os.path.join(module_path, "configuration_v1_1.py"))
        config_mod = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_mod)
        Configuration = config_mod.Configuration
        get_config_instance = config_mod.get_config_instance
        
        print("Absolute imports successful")
    except Exception as e:
        print(f"Error importing modules: {e}")
        raise

# Default model
model = init_chat_model("ollama:llama3.2", temperature=0.2)

current_date = datetime.now().strftime("%A, %B %d, %Y")

MAX_SEARCH_RETRIES = 5

# Default database configuration
DEFAULT_DATABASE = "Qwen--Qwen3-Embedding-0.6B--3000--600"
DEFAULT_TENANT = "2025-04-22_15-41-10"


class GraphState(MessagesState):
    original_question: str
    attempted_search_queries: list[str]
    database_path: str = None
    tenant_id: str = None


relevance_evaluator = create_async_llm_as_judge(
    judge=model,
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {current_date}.",
    feedback_key="retrieval_relevance",
)

helpfulness_evaluator = create_async_llm_as_judge(
    judge=model,
    prompt=RAG_HELPFULNESS_PROMPT
    + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {current_date}.',
    feedback_key="helpfulness",
)


SYSTEM_PROMPT = """
Use the provided local database retrieval tool to find information relevant to the user's question.
"""


# Create a local database retrieval tool
@tool
async def local_retrieval_tool(query: str, database_path: str = None, tenant_id: str = None):
    """Search the local database for information relevant to the query."""
    # Set default values if not provided
    if database_path is None:
        # Use the default database path
        database_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "local-rag-researcher-deepseek-he",
            "database",
            DEFAULT_DATABASE
        )
    
    if tenant_id is None:
        tenant_id = DEFAULT_TENANT
    
    # Use the search_documents function from vector_db_v1_1.py
    try:
        # Default to English language for retrieval
        documents = search_documents(query=query, k=5, language="English")
        
        # Format the documents for better readability
        formatted_docs = format_documents_as_plain_text(documents)
        
        # Return the formatted documents
        return {"results": formatted_docs}
    except Exception as e:
        return {"error": str(e), "results": "No documents found."}


model_with_tools = model.bind_tools([local_retrieval_tool])


async def relevance_filter(state: GraphState):
    """Filter out irrelevant search results."""
    query = state["original_question"]
    search_results = state["messages"][-1].content
    is_relevant = await relevance_evaluator.ainvoke({"query": query, "search_results": search_results})
    if is_relevant == "true":
        return {"messages": state["messages"]}
    return {}


async def should_continue(state: GraphState):
    if len(state["attempted_search_queries"]) > MAX_SEARCH_RETRIES:
        return END
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "local_retrieval"
    return "reflect"


async def call_model(state: GraphState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = await model_with_tools.ainvoke(messages)
    if response.tool_calls and response.tool_calls[0]["name"] == local_retrieval_tool.name:
        search_query = response.tool_calls[0]["args"]["query"]
        
        # Extract database_path and tenant_id from tool_calls if provided
        tool_args = response.tool_calls[0]["args"]
        database_path = tool_args.get("database_path", state.get("database_path"))
        tenant_id = tool_args.get("tenant_id", state.get("tenant_id"))
        
        return {
            "messages": [response],
            "attempted_search_queries": state["attempted_search_queries"] + [search_query],
            "database_path": database_path,
            "tenant_id": tenant_id,
        }
    return {"messages": [response]}


async def local_retrieval(state: GraphState):
    last_message = state["messages"][-1]
    
    # Extract the query and optional database parameters
    tool_args = last_message.tool_calls[0]["args"]
    query = tool_args["query"]
    database_path = tool_args.get("database_path", state.get("database_path"))
    tenant_id = tool_args.get("tenant_id", state.get("tenant_id"))
    
    # Call the local retrieval tool with the appropriate parameters
    # Fix: Pass parameters as a single input dictionary as required by StructuredTool.ainvoke()
    search_results = await local_retrieval_tool.ainvoke(input={
        "query": query,
        "database_path": database_path,
        "tenant_id": tenant_id
    })
    
    return {"messages": [search_results]}


async def reflect(state: GraphState):
    """Reflect on the answer and decide whether to retry."""
    question = state["original_question"]
    answer = state["messages"][-1].content

    # Evaluate the helpfulness of the answer
    is_helpful = await helpfulness_evaluator.ainvoke({"question": question, "answer": answer})

    if is_helpful == "true":
        # If the answer is helpful, we're done
        return {"messages": state["messages"]}

    # If we've already tried the maximum number of times, we're done
    if len(state["attempted_search_queries"]) >= MAX_SEARCH_RETRIES:
        # Add a reflection message
        reflection_message = {
            "role": "assistant",
            "content": "I apologize, but I'm having trouble finding relevant information to answer your question accurately. Let me provide the best answer I can based on my general knowledge.",
        }
        return {"messages": state["messages"] + [reflection_message]}

    # Otherwise, try again with a different approach
    reflection_message = {
        "role": "assistant",
        "content": f"""
I originally asked you the following question:

<original_question>
{state["original_question"]}
</original_question>

Your answer was not helpful for the following reason:

<reason>
The answer was not relevant to the question.
</reason>

Please check the conversation history carefully and try again. You may choose to fetch more information if you think the answer
to the original question is not somewhere in the conversation, but carefully consider if the answer is already in the conversation.

You have already attempted to answer the original question using the following search queries,
so if you choose to search again, you must rephrase your search query to be different from the ones below to avoid fetching redundant information:

<attempted_search_queries>
{state['attempted_search_queries']}
</attempted_search_queries>

As a reminder, check the previous conversation history and fetched context carefully before searching again!
""",
        }
    
    return {"messages": state["messages"] + [reflection_message]}


async def retry_or_end(state: GraphState):
    if state["messages"][-1].type == "human":
        return "agent"
    return END


async def store_database_config(state: GraphState):
    """Store the database configuration in the state."""
    # Get the database path and tenant ID from the parameters or use defaults
    database_path = state.get("database_path")
    tenant_id = state.get("tenant_id")
    
    # If not provided, use the defaults
    if database_path is None:
        database_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "local-rag-researcher-deepseek-he",
            "database",
            DEFAULT_DATABASE
        )
    
    if tenant_id is None:
        tenant_id = DEFAULT_TENANT
    
    return {
        "database_path": database_path,
        "tenant_id": tenant_id
    }


workflow = StateGraph(GraphState, input=MessagesState, output=MessagesState)

workflow.add_node(
    "store_original_question",
    lambda state: {
        "original_question": state["messages"][-1].content,
        "attempted_search_queries": [],
    },
)
workflow.add_node("store_database_config", store_database_config)
workflow.add_node("agent", call_model)
workflow.add_node("local_retrieval", local_retrieval)
workflow.add_node("relevance_filter", relevance_filter)
workflow.add_node("reflect", reflect)

workflow.add_edge(START, "store_original_question")
workflow.add_edge("store_original_question", "store_database_config")
workflow.add_edge("store_database_config", "agent")
workflow.add_conditional_edges("agent", should_continue, ["local_retrieval", "reflect", END])
workflow.add_edge("local_retrieval", "relevance_filter")
workflow.add_edge("relevance_filter", "agent")
workflow.add_conditional_edges(
    "reflect",
    retry_or_end,
    ["agent", END],
)

agent = workflow.compile()


# Function to initialize the agent with custom database configuration
def init_agent(database_path=None, tenant_id=None):
    """
    Initialize the corrective RAG agent with custom database configuration.
    
    Args:
        database_path (str, optional): Path to the database directory.
        tenant_id (str, optional): Tenant ID for the database.
        
    Returns:
        The initialized agent.
    """
    # The agent is already compiled, but we can set the initial state
    # when we invoke it later
    return agent


# Example usage
async def run_agent(question, database_path=None, tenant_id=None):
    """
    Run the agent with a question and optional database configuration.
    
    Args:
        question (str): The question to ask the agent.
        database_path (str, optional): Path to the database directory.
        tenant_id (str, optional): Tenant ID for the database.
        
    Returns:
        The agent's response.
    """
    # Create initial state with database configuration
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "database_path": database_path,
        "tenant_id": tenant_id,
    }
    
    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    # Return the last message from the agent
    return result["messages"][-1].content
