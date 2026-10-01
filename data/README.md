# Data

This directory holds the inputs and outputs of the RAG pipeline. Paths are relative to the repository root (`/app` in the container). The `DATA_DIR` and `CHROMA_DIR` settings move the corpus and the index (see `.env.example`).

| Path | Contents | In git | Produced by |
|---|---|---|---|
| `raw/` | Source corpus (`DATA_DIR`) | Yes, if the licenses allow it | You, or a download command (Phase 2) |
| `eval/` | Evaluation questions and the committed result files | Yes | Questions written by hand; results from the evaluation and load-test runs (Phases 7–8) |
| `chroma_db/` | Persistent Chroma vector index (`CHROMA_DIR`) | No (gitignored) | `agentic-rag ingest` |

## `raw/`: the corpus

These are the documents the chatbot answers from. The domain and the corpus are chosen in Phase 2 (plan decision 8). Until then the directory holds only `.gitkeep`.

- **Small and well processed.** A few dozen clean documents work better than a large, noisy crawl. Every document should be on topic, have a real text layer (no scanned PDFs without one), and stay stable enough for the reference answers of the evaluation set.
- **Licensing.** Commit a document only if its license allows redistribution, such as public domain, Creative Commons or your own text. Otherwise commit no copy: provide a download command instead (Phase 2) and record the source URL. In both cases, list every document in `raw/README.md` with its source URL, license and retrieval date.
- **Formats.** They are decided in Phase 2 together with the corpus. The plan names PDF, Markdown and plain text.
- **What gets indexed.** The loaders read every file under `raw/`, recursively. They skip:
  - dotfiles and files inside dot-directories, such as `.gitkeep`;
  - README files (`README`, `README.md`, `README.hu.md`, in any letter case).

  This is why the source list can live in `raw/README.md`, next to the documents, without being indexed.
- **Citations.** Each document carries this metadata, which the UI shows and the answers cite. Choose meaningful file and directory names, because they appear in the UI.
  - `source`: the path relative to `raw/`;
  - `title`;
  - `page` (1-based), where the format has pages;
  - `section`, where the format has headings.
- **Container.** The image contains no corpus. Compose mounts `./data/raw` read-only at `/app/data/raw`.

## `eval/`: the evaluation set

This directory holds the functional evaluation set, `questions.jsonl`: 10–20 questions with reference answers and expected sources. Its `results/` directory holds the committed outputs of the final evaluation and load-test runs. See [`eval/README.md`](eval/README.md).

## `chroma_db/`: the vector index

The index is generated and never edited by hand. It is gitignored by the `chroma_db/` rule in `.gitignore`.

- **Build it** with `agentic-rag ingest` (or `python -m agentic_rag ingest`). Every run makes the index match the corpus: it upserts the chunks of every file, then deletes the stored chunks that the run did not produce. A plain `ingest` therefore picks up added, edited, shortened and removed files and changes to the chunking, and running it twice changes nothing. Ingestion is implemented in Phase 2; until then the command only reports that it is planned.
- **Rebuild it** with `agentic-rag ingest --rebuild` after you change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`. This includes switching between the offline fake embeddings and the Hugging Face model. `--rebuild` discards the index and builds it from scratch, because vectors of different models are not comparable. From Phase 2 on, the index records the provider and the model it was built with, so a plain `ingest` or a query with another one fails instead of mixing the vectors of two models or returning meaningless matches.
- **Container.** The index lives in the named volume `chroma-data`, mounted at `/app/data/chroma_db`, so it survives image rebuilds. `docker compose down -v` deletes it. With `INGEST_ON_START=true`, the container builds the index at start-up when it is missing (Phase 6).
