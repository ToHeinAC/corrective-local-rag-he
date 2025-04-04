from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT

llm = init_chat_model("ollama:qwen2.5:7b", temperature=0.1)

tavily_tool = TavilySearch(max_results=10)

current_date = datetime.now().strftime("%A, %B %d, %Y")

relevance_evaluator = create_llm_as_judge(
    judge=llm,
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {current_date}.",
    feedback_key="retrieval_relevance",
)

agent = create_react_agent(
    llm,
    prompt=f"The current date is {current_date}.",
    tools=[tavily_tool],
)
