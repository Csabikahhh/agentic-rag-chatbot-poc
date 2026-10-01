"""Ingestion pipeline: load the corpus, split it into chunks, embed and store them in Chroma.

Load -> split -> embed -> store (plan section 5.4), one module per step:

- ``loaders``: the files in ``DATA_DIR`` become LangChain ``Document`` objects with citation
  metadata (``source``, ``title``, ``page``, ``section``).
- ``chunking``: the splitter configuration and the metadata every chunk adds (``chunk_id``).
- ``index``: build and open the persistent Chroma index in ``CHROMA_DIR``; ``build_index``
  backs the ``agentic-rag ingest`` command.

The embedding model comes from ``agentic_rag.embeddings.get_embeddings``, the same for
building and for querying the index. The modules import their heavy libraries (chromadb,
langchain_chroma, langchain_text_splitters, sentence-transformers) inside the functions that
need them, so importing the package stays fast.
"""
