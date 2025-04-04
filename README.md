# OpenEvals Local RAG

This repo contains an implementation of effective Retrieval-Augmented Generation (RAG) over web search results using a small, fast, local LLM (Qwen-2.5 7b) through [Ollama](https://ollama.com) and [Tavily](https://tavily.com/)'s search engine tool.

It shows off the use of [OpenEvals](https://github.com/langchain-ai/openevals) RAG evaluators "in-the-loop" as part of the agent. Compared to a naive approach of stuffing all web results back into LLM context, this improves performance in two ways:

- A [corrective RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)-inspired approach of grading and filtering retrieved search results for relevancy.
  - This reduces the amount of distracting information the small LLM has to deal with when generating a final response.
- A final `helpfulness` evaluator run as a reflection step over generated answers. If an answer is not deemed helpful in answering the original question (stored in the `store_original_question` step in the diagram below), the LLM is reprompted to try again.
  - This ensures that the small LLM hasn't lost track of its original goal, which can happen if retrieval returns many relevant results.

![](/static/img/corrective_rag.png)

The thought behind running evaluators as part of your agent is to proactively attempt to fix errors - if you have some defined metrics that you want parts of your agent to succeed on and your evaluators return suggestions, your agent can course-correct based on that feedback and improve its performance. To show this, compare the following two traces where the LLM was asked about a basketball team's record in the previous season:

- [Using basic ReAct architecture](https://smith.langchain.com/public/b4dbe71f-062f-4a19-a11b-096cefcb630c/r), the LLM presents a rambling answer about historical records while omitting the relevant year.
- [Using the enhanced architecture](https://smith.langchain.com/public/c301728d-0b20-4d1d-8601-9c02727930bb/r), the LLM remains focused and correctly uses retrieved results to generate a correct answer.

## Getting started

You will need:
- Python 3.11 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- The `uv` package manager (recommended)

1. Clone the repository:

```bash
git clone https://github.com/jacoblee93/openevals-local-rag.git
cd openevals-local-rag
```

2. Set environment variables

First, rename the existing `.env.example` file in the cloned repo to `.env`. Next, go to [Tavily's website](https://tavily.com/) and sign up for an API key and set it as `TAVILY_API_KEY`.

Your `.env` file should look like this

```
TAVILY_API_KEY=YOUR_KEY_HERE
```

3. Install Ollama

This project uses [Ollama](https://ollama.com/) to run Qwen locally. Follow their instructions to download and install it.

Then, you will need run the following command to download and run [`qwen2.5:7b`](https://ollama.com/library/qwen2.5:7b):

```bash
ollama run qwen2.5:7b
```

4. Install dependencies

This repo is set up to use [uv](https://docs.astral.sh/uv/). Run `uv sync` to install require deps:

```bash
uv sync

# Or, if you don't have uv installed and don't want to use it:
# pip install
```

5. (Optional) Sign up for LangSmith

If you want to run this project using LangGraph Studio, you will need to sign up for [LangSmith](https://smith.langchain.com). Otherwise, you will need to run the included agents by importing them as modules.

6. Run your agent

You can now run your agent using the following command:

```bash
uv run langgraph dev
```

Once Studio loads, you can try both the simpler `react_agent` and the improved `corrective_rag_agent` by switching the toggle in the top left:

![](/static/img/studio_toggle.png)

Now try a few queries like `What record did the Warriors have last year?` or `What do modern historians say about the fall of the Roman Empire?` and see how the results differ!

## Thank you!

This repo is meant to provide inspiration for how running evaluators "in-the-loop" as part of your agent can help improve performance. If you have questions or comments, please open an issue or reach out to us [@LangChainAI](https://x.com/langchainai) on X (formerly Twitter)!
