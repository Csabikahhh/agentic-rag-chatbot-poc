"""Streamlit user interface of the chatbot (plan section 5.5).

Modules:

- ``app``: the entrypoint script, run with ``streamlit run src/agentic_rag/ui/app.py`` from
  the repository root (``/app`` in the container). It is executed by Streamlit, never imported.
- ``components``: the building blocks of the page: the agent-step panel, the retrieved-context
  panel, the configuration summary and the chat-turn record kept in the session state.
"""
