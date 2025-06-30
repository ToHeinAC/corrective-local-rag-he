import json
import asyncio
import os
import sys
import importlib.util
from datetime import datetime
from types import ModuleType

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool

# Create a dummy tavily module to prevent import errors
class DummyTavilyClient:
    """Dummy implementation of TavilyClient to prevent import errors"""
    def __init__(self, *args, **kwargs):
        pass
        
    def search(self, *args, **kwargs):
        print("DummyTavilyClient.search called")
        return []

# Create dummy tavily module
dummy_tavily = ModuleType('tavily')
dummy_tavily.TavilyClient = DummyTavilyClient

# Add dummy module to sys.modules if tavily is not installed
if 'tavily' not in sys.modules:
    print("Adding dummy tavily module to sys.modules")
    sys.modules['tavily'] = dummy_tavily

# Custom implementation of evaluators since openevals is not available
from langchain.prompts import PromptTemplate

# Define prompts for evaluators
RAG_RETRIEVAL_RELEVANCE_PROMPT = """
You are an objective judge evaluating the relevance of search results to a user's query.

<query>
{query}
</query>

<search_results>
{search_results}
</search_results>

Based on the search results provided, are they relevant to the query?
Return ONLY 'true' if the results are relevant to the query, or 'false' if they are not relevant.
"""

RAG_HELPFULNESS_PROMPT = """
You are an objective judge evaluating the helpfulness of an answer to a user's question.

<question>
{question}
</question>

<answer>
{answer}
</answer>

Based on the answer provided, is it helpful to the user's question?
"""

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
DEFAULT_TENANT = "default"


class GraphState(MessagesState):
    original_question: str
    attempted_search_queries: list[str]
    database_path: str = None
    tenant_id: str = None


# Create prompt templates for the evaluators
relevance_prompt_template = PromptTemplate(
    template=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {current_date}.",
    input_variables=["query", "search_results"]
)

helpfulness_prompt_template = PromptTemplate(
    template=RAG_HELPFULNESS_PROMPT + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {current_date}.',
    input_variables=["question", "answer"]
)

# Implement async evaluator functions
async def relevance_evaluator(inputs):
    """
    Evaluate the relevance of search results to a query.
    
    Args:
        inputs: A dictionary with keys 'query' and 'search_results'
        
    Returns:
        'true' if the results are relevant, 'false' otherwise
    """
    try:
        query = inputs["query"]
        search_results = inputs["search_results"]
        
        # Format the prompt with the inputs
        formatted_prompt = relevance_prompt_template.format(
            query=query,
            search_results=search_results
        )
        
        # Use the model to evaluate relevance
        result = await model.ainvoke(formatted_prompt)
        result_text = result.content.strip().lower()
        
        # Extract just 'true' or 'false' from the response
        if "true" in result_text:
            return "true"
        else:
            return "false"
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
        # Default to true in case of error
        return "true"

async def helpfulness_evaluator(inputs):
    """
    Evaluate the helpfulness of an answer to a question.
    
    Args:
        inputs: A dictionary with keys 'question' and 'answer'
        
    Returns:
        'true' if the answer is helpful, 'false' otherwise
    """
    try:
        question = inputs["question"]
        answer = inputs["answer"]
        
        # Format the prompt with the inputs
        formatted_prompt = helpfulness_prompt_template.format(
            question=question,
            answer=answer
        )
        
        # Use the model to evaluate helpfulness
        result = await model.ainvoke(formatted_prompt)
        result_text = result.content.strip().lower()
        
        # Extract just 'true' or 'false' from the response
        if "true" in result_text:
            return "true"
        else:
            return "false"
    except Exception as e:
        print(f"Error in helpfulness evaluation: {e}")
        # Default to true in case of error
        return "true"


SYSTEM_PROMPT = """
Use the provided local database retrieval tool to find information relevant to the user's question.
"""


# Function to detect language of text
def detect_language(text):
    """Detect the language of the input text.
    
    This is a simple implementation that checks for common German characters.
    For a production system, consider using a proper language detection library like langdetect.
    """
    # Check for common German characters/words
    german_chars = ['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü']
    german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'für', 'mit']
    
    # Convert to lowercase for word matching
    text_lower = text.lower()
    
    # Check for German characters
    for char in german_chars:
        if char in text:
            return "German"
    
    # Check for common German words
    for word in german_words:
        if f" {word} " in f" {text_lower} ":
            return "German"
    
    # Default to English if no German indicators found
    return "English"

# Create a local database retrieval tool
@tool
async def local_retrieval_tool(query: str, database_path: str = None, tenant_id: str = None, language: str = None):
    """Search the local database for information relevant to the query.
    
    Args:
        query: The search query
        database_path: Optional path to the database directory
        tenant_id: Optional tenant ID for the database
        language: Optional language of the query (English or German)
    """
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
    
    # Detect language if not provided
    if language is None:
        language = detect_language(query)
        print(f"Detected language: {language} for query: {query}")
    
    # Use the search_documents function from vector_db_v1_1.py
    try:
        print(f"Using database_path: {database_path}, tenant_id: {tenant_id}, language: {language}")
        
        # First, check if the tavily module is required and handle its absence gracefully
        try:
            # Try to import necessary modules
            from src.assistant.v1_1.vector_db_v1_1 import search_documents as vector_search
            print("Successfully imported search_documents function")
        except ImportError as e:
            if "tavily" in str(e):
                print("Tavily module not found. Using local search only.")
                # Create a simplified search function that doesn't depend on tavily
                def vector_search(query, k=5, language="English"):
                    # Simple implementation that doesn't use tavily
                    print(f"Performing local search for: {query} in language: {language}")
                    # Return empty results since we can't do the actual search without dependencies
                    return []
            else:
                # Re-raise if it's a different import error
                raise
        
        # Override the default database path and tenant ID in the search_documents function
        # by modifying the imported module's constants
        import sys
        vector_db_module = sys.modules.get('src.assistant.v1_1.vector_db_v1_1')
        
        # Store original values to restore later
        original_db_path = None
        original_tenant_id = None
        
        if vector_db_module and database_path:
            if hasattr(vector_db_module, 'VECTOR_DB_PATH'):
                original_db_path = vector_db_module.VECTOR_DB_PATH
                vector_db_module.VECTOR_DB_PATH = database_path
                print(f"Set VECTOR_DB_PATH to: {database_path}")
        
        if vector_db_module and tenant_id:
            if hasattr(vector_db_module, 'DEFAULT_TENANT_ID'):
                original_tenant_id = vector_db_module.DEFAULT_TENANT_ID
                vector_db_module.DEFAULT_TENANT_ID = tenant_id
                print(f"Set DEFAULT_TENANT_ID to: {tenant_id}")
        
        # Use detected or specified language for retrieval
        documents = vector_search(query=query, k=5, language=language)
        
        # Restore original values
        if vector_db_module:
            if original_db_path is not None and hasattr(vector_db_module, 'VECTOR_DB_PATH'):
                vector_db_module.VECTOR_DB_PATH = original_db_path
            if original_tenant_id is not None and hasattr(vector_db_module, 'DEFAULT_TENANT_ID'):
                vector_db_module.DEFAULT_TENANT_ID = original_tenant_id
        
        # Format the documents for better readability
        if documents:
            formatted_docs = format_documents_as_plain_text(documents)
        else:
            formatted_docs = "No documents found for the query."
        
        # Return the formatted documents
        return {"results": formatted_docs, "detected_language": language, "database_path": database_path, "tenant_id": tenant_id}
    except Exception as e:
        print(f"Error in local_retrieval_tool: {e}")
        return {"error": str(e), "results": "No documents found.", "detected_language": language, "database_path": database_path, "tenant_id": tenant_id}


model_with_tools = model.bind_tools([local_retrieval_tool])


async def relevance_filter(state: GraphState):
    """Filter out irrelevant search results."""
    query = state["original_question"]
    search_results = state["messages"][-1].content
    
    # Instead of using ainvoke, directly call the relevance_evaluator function
    try:
        # Check if relevance_evaluator is callable directly
        if callable(relevance_evaluator):
            is_relevant = await relevance_evaluator({"query": query, "search_results": search_results})
        else:
            # Fallback: assume all results are relevant if we can't evaluate
            print("Warning: relevance_evaluator is not callable, assuming results are relevant")
            is_relevant = "true"
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
        # Fallback: assume all results are relevant if evaluation fails
        is_relevant = "true"
    
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
    language = tool_args.get("language", None)
    
    try:
        # Call the local retrieval tool with the appropriate parameters
        # Fix: Pass parameters as a single input dictionary as required by StructuredTool.ainvoke()
        search_results = await local_retrieval_tool.ainvoke(input={
            "query": query,
            "database_path": database_path,
            "tenant_id": tenant_id,
            "language": language
        })
        
        # Format the response as a proper message with role and content
        if isinstance(search_results, dict) and "error" in search_results:
            # Format error message properly
            error_message = {
                "role": "assistant",
                "content": f"Error during retrieval: {search_results['error']}\nNo documents found. Please try a different query or database configuration."
            }
            return {"messages": [error_message]}
        else:
            # Format successful response properly
            return {"messages": [{
                "role": "assistant",
                "content": search_results.get("results", "No results found.")
            }]}
    except Exception as e:
        # Handle any exceptions and format as proper message
        error_message = {
            "role": "assistant",
            "content": f"Error during retrieval: {str(e)}\nNo documents found. Please try a different query or database configuration."
        }
        return {"messages": [error_message]}



async def reflect(state: GraphState):
    """Reflect on the answer and decide whether to retry."""
    question = state["original_question"]
    answer = state["messages"][-1].content

    # Evaluate the helpfulness of the answer
    try:
        # Check if helpfulness_evaluator is callable directly
        if callable(helpfulness_evaluator):
            is_helpful = await helpfulness_evaluator({"question": question, "answer": answer})
        else:
            # Fallback: assume answer is helpful if we can't evaluate
            print("Warning: helpfulness_evaluator is not callable, assuming answer is helpful")
            is_helpful = "true"
    except Exception as e:
        print(f"Error in helpfulness evaluation: {e}")
        # Fallback: assume answer is helpful if evaluation fails
        is_helpful = "true"

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
