# Data

This directory holds the inputs and outputs of the RAG pipeline. Paths are relative to the repository root (`/app` in the container). The `DATA_DIR` and `CHROMA_DIR` settings move the corpus and the index (see `.env.example`).

| Path | Contents | In git | Produced by |
|---|---|---|---|
| `sources.toml` | The corpus sources: repository, pinned commit, patterns, license, name and URL template of each | Yes | Written by hand |
| `raw/` | Source corpus (`DATA_DIR`), one directory per source | No (gitignored), it is downloaded | `agentic-rag ingest --download`, from the sources in `sources.toml` |
| `eval/` | Evaluation questions and the committed result files | Yes | Questions written by hand; results from the evaluation and load-test runs (Phases 7–8) |
| `chroma_db/` | Persistent Chroma vector index (`CHROMA_DIR`) | No (gitignored) | `agentic-rag ingest` |

## `raw/`: the corpus

These are the documents the chatbot answers from: the official documentation of MDN Web Docs (a curated subset), React, Vue, Next.js, Nuxt and the TypeScript Handbook (plan decision 8), 1 160 pages. `agentic-rag ingest --download` puts them here; in git the directory holds only `.gitkeep`.

- **Small and well processed.** A curated subset works better than a complete copy: the current guides and API references of each source, without the legacy and migration sections. Every document should be on topic and, pinned to a commit, stay stable enough for the reference answers of the evaluation set.
- **Licensing.** Commit a document only if its license allows redistribution, such as public domain, Creative Commons or your own text. Otherwise commit no copy: provide a download command instead and record the source. The chosen documentation is downloaded, not committed, which also keeps the share-alike text of MDN (CC BY-SA 2.5) out of the repository; `sources.toml` records the repository, the pinned commit, the patterns and the license of every source.
- **One directory per source.** The download copies each source into `raw/<id>/` (`mdn/`, `react/`, `vue/`, `nextjs/`, `nuxt/`, `typescript/`) and writes the source's entry there as `.source.json`, the manifest. The loaders read the manifest to add the source's name to every page title and to build each page's public URL, so the corpus describes itself also in the container, where only `raw/` is mounted. A source whose manifest matches `sources.toml` is not downloaded again; to force a download, delete its directory.
- **Data for a tool.** `raw/browser-compat-data/` holds MDN's browser compatibility tables (CC0), which the `browser_support` tool reads. Its entry in `sources.toml` has `index = false`, so the loaders skip it: it is downloaded with the corpus but never indexed.
- **Formats.** Markdown and MDX, the source formats of the documentation repositories, and plain `.txt`; any other file stops the ingestion. The loaders remove the front matter (keeping the title), MDN's `{{macro}}` calls, the JSX, VitePress and MDC components and the HTML tags, keep the code blocks verbatim, and split every page at its H2 and H3 headings (see `src/agentic_rag/ingestion/markdown.py`).
- **What gets indexed.** The loaders read every file under `raw/`, recursively. They skip:
  - dotfiles and files inside dot-directories, such as `.gitkeep`;
  - README files (`README`, `README.md`, `README.hu.md`, in any letter case).

  This is why a README can describe the documents next to them without being indexed, and why the README files of the downloaded repositories are left out.
- **Citations.** Each document carries this metadata, which the UI shows and the answers cite:
  - `source`: the path relative to `raw/`, starting with the source's directory;
  - `title`: the page title and the source's name, for example `useState – React`;
  - `section`: the H2 and H3 headings, for example `Usage > Adding state to a component`;
  - `url`: the page on the documentation site;
  - `page` (1-based), for paged formats, which this corpus does not have.
- **Container.** The image contains no corpus, only `sources.toml`. At start-up the container downloads the corpus into the named volume `corpus-data`, mounted at `/app/data/raw` (`agentic-rag serve` with `INGEST_ON_START`); the host's `data/raw` is not mounted.

## `eval/`: the evaluation set

This directory holds the functional evaluation set, `questions.jsonl`: 10–20 questions with reference answers and expected sources. Its `results/` directory holds the committed outputs of the final evaluation and load-test runs. See [`eval/README.md`](eval/README.md).

## `chroma_db/`: the vector index

The index is generated and never edited by hand. It is gitignored by the `chroma_db/` rule in `.gitignore`.

- **Build it** with `agentic-rag ingest` (or `python -m agentic_rag ingest`); `ingest --download` first downloads the corpus. Every run makes the index match the corpus: it embeds and stores the chunks the index does not hold yet, then deletes the stored chunks that the run did not produce. A plain `ingest` therefore picks up added, edited, shortened and removed files and changes to the chunking, and running it twice changes nothing: an unchanged corpus embeds nothing (8 s for the whole corpus, against about 6 minutes for the first build on the CPU). A corpus without documents stops the command instead of emptying the index.
- **Rebuild it** with `agentic-rag ingest --rebuild` after you change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`. This includes switching between the offline fake embeddings and the Hugging Face model. `--rebuild` discards the index and builds it from scratch, because vectors of different models are not comparable. The index records the provider and the model it was built with, so a plain `ingest` or a query with another one fails (`EmbeddingMismatchError`, exit code 2) instead of mixing the vectors of two models or returning meaningless matches.
- **Container.** The index lives in the named volume `chroma-data`, mounted at `/app/data/chroma_db`, in one directory per embedding provider (`CHROMA_DIR=/app/data/chroma_db/<EMBEDDING_PROVIDER>`), so it survives image rebuilds and switching between the full stack and fake mode. The container builds it at start-up when it is missing and brings it up to date otherwise; `docker compose run --rm --no-deps app agentic-rag ingest --rebuild` rebuilds it. `docker compose down -v` deletes it.
