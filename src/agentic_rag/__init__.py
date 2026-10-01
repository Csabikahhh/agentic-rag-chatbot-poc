"""Agentic RAG chatbot prototype.

A LangGraph main workflow that plans, routes and executes sub-tasks, a modular RAG
subgraph over a local vector index, a local open-source LLM served by Ollama (or a
scripted fake for tests) and a Streamlit UI that shows the agent's steps.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentic-rag-chatbot-poc")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
