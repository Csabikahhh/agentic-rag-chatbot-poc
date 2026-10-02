# Data

This directory holds the inputs and outputs of the RAG pipeline. Paths are relative to the repository root (`/app` in the container). The `DATA_DIR` and `CHROMA_DIR` settings move the corpus and the index (see `.env.example`).

| Path | Contents | In git | Produced by |
|---|---|---|---|
| `sources.yaml` | The corpus sources: repository, pinned commit, paths and license of each | Yes | Written by hand (Phase 2) |
| `raw/` | Source corpus (`DATA_DIR`) | No, it is downloaded | `agentic-rag ingest --download`, from the sources in `sources.yaml` (Phase 2) |
| `eval/` | Evaluation questions and the committed result files | Yes | Questions written by hand; results from the evaluation and load-test runs (Phases 7–8) |
| `chroma_db/` | Persistent Chroma vector index (`CHROMA_DIR`) | No (gitignored) | `agentic-rag ingest` |

## `raw/`: the corpus

These are the documents the chatbot answers from: the official documentation of MDN Web Docs (a curated subset), React, Vue, Next.js, Nuxt and the TypeScript Handbook (plan decision 8). Phase 2 downloads them; until then the directory holds only `.gitkeep`.

- **Small and well processed.** A curated subset works better than a complete copy: the current guides and API references of each source, without the legacy and migration sections. Every document should be on topic and, pinned to a commit, stay stable enough for the reference answers of the evaluation set.
- **Licensing.** Commit a document only if its license allows redistribution, such as public domain, Creative Commons or your own text. Otherwise commit no copy: provide a download command instead and record the source. The chosen documentation is downloaded, not committed, which also keeps the share-alike text of MDN (CC BY-SA 2.5) out of the repository; `sources.yaml` records the repository, the pinned commit, the paths and the license of every source.
- **Formats.** Markdown and MDX, the source formats of the documentation repositories. The loaders strip the front matter (keeping the title), MDN's `{{macro}}` calls and the MDX components (Phase 2).
- **What gets indexed.** The loaders read every file under `raw/`, recursively. They skip:
  - dotfiles and files inside dot-directories, such as `.gitkeep`;
  - README files (`README`, `README.md`, `README.hu.md`, in any letter case).

  This is why a README can describe the documents next to them without being indexed, and why the README files of the downloaded repositories are left out.
- **Citations.** Each document carries this metadata, which the UI shows and the answers cite. Every source is downloaded into a directory of its own (`mdn/`, `react/`, `vue/`, `nextjs/`, `nuxt/`, `typescript/`), so the path shows which documentation a citation comes from.
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
