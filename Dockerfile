# syntax=docker/dockerfile:1

# Image for the Streamlit UI of the agentic RAG chatbot prototype.
#
#   Build: docker build -t agentic-rag-chatbot:dev .
#   Run:   see compose.yaml (full stack with Ollama, or model-free fake mode), or
#          docker run --rm -p 127.0.0.1:8501:8501 \
#            -e LLM_PROVIDER=fake -e EMBEDDING_PROVIDER=fake agentic-rag-chatbot:dev
#
# The command, agentic-rag serve, first prepares the knowledge base (INGEST_ON_START):
# it downloads the corpus into /app/data/raw and builds the index in
# /app/data/chroma_db, then starts the UI. Mount volumes on both paths to keep
# them between containers; without volumes every new container downloads and
# builds them again.
#
# The app runs as UID/GID 10001. On Linux a bind mount keeps the host's owner, so
# if the app must write to one, build the image with your own IDs:
#   docker build --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" -t agentic-rag-chatbot:dev .
# With Compose: APP_UID=$(id -u) APP_GID=$(id -g) docker compose up --build
#
# No API keys or other credentials are baked in: configuration comes from
# environment variables at run time (see .env.example).
#
# Portions adapted from the asset Dockerfile.simple of the
# docker-project-foundations skill (https://github.com/docker/skills), licensed
# under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). Modified for this project:
# rewritten for a uv-based Python build.

# The deps stage and the runtime must use the same Python image: the virtual
# environment links to the interpreter at /usr/local/bin.
ARG PYTHON_IMAGE=python:3.12.14-slim-trixie

# uv pinned to the version that wrote uv.lock. The RUN steps mount it, so no
# image layer contains it.
FROM ghcr.io/astral-sh/uv:0.12.6 AS uv

# ---------------------------------------------------------------------------
# Stage 1: the dependencies pinned in uv.lock, without the project itself
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS deps

# Copy instead of hard-linking from the cache mount, and never download a
# Python: the venv must use the interpreter of the shared base image. No
# bytecode here (see the runtime stage): without it, the files of the venv come
# out the same each time this step runs with the same uv.lock. Only their
# timestamps differ, and the cache key of the runtime stage's COPY ignores them.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# --locked stops the build when uv.lock is out of date with pyproject.toml.
# torch comes from the PyTorch CPU index, as locked in uv.lock, so no CUDA
# wheels are installed.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 2: runtime image without uv, build files or dev dependencies
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# git downloads the corpus at start-up (sparse checkouts of the pinned commits).
# Installed first, so its layer is reused by every rebuild.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="agentic-rag-chatbot-poc" \
      org.opencontainers.image.description="Agentic RAG chatbot prototype: LangGraph workflow, RAG subgraph, local LLM via Ollama, Streamlit UI" \
      org.opencontainers.image.source="https://github.com/Csabikahhh/agentic-rag-chatbot-poc" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# The two big layers (the venv and its bytecode) come first and change only
# when the deps stage produces a different venv: edits of src/, README.md or
# LICENSE rebuild only the small layers further down. When the deps stage runs
# again with the same result (after a pyproject.toml edit that leaves the
# dependencies alone), BuildKit reuses both layers.
# Code and dependencies stay root-owned, so the app cannot modify them.
COPY --link --from=deps /app/.venv /app/.venv

# Bytecode, compiled at build time because the venv is read-only for the app
# user. uv's own compilation is not reproducible (a few files differ between
# runs), so it would change the venv layer above whenever the deps stage runs.
RUN /app/.venv/bin/python -m compileall -q -j 0 /app/.venv/lib

# Numeric IDs, so orchestrators can verify that the user is not root.
# compose.yaml passes APP_UID and APP_GID (default 10001). Declared after the
# big layers, so other IDs rebuild only the small layers below. --non-unique
# accepts IDs that the base image already uses, such as GID 20 or 100.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --non-unique --gid "${APP_GID}" app \
    && useradd --non-unique --uid "${APP_UID}" --gid "${APP_GID}" --no-log-init \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app

# HF_HOME is the mount point of the Hugging Face cache volume (compose.yaml).
# Streamlit: headless, no usage statistics, no file watcher (the code in the
# image does not change), and a fixed browser address, which also skips the
# external-IP lookup (an outbound HTTP request) that it does at start-up.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_SERVER_ADDRESS=localhost

# Writable paths for the app user. Named volumes mounted on them (corpus, Chroma
# index, Hugging Face cache) are initialised with this ownership. Keep the paths
# in sync with compose.yaml.
RUN mkdir -p /app/data/raw /app/data/chroma_db "${HF_HOME}" \
    && chown "${APP_UID}:${APP_GID}" /app/data /app/data/raw /app/data/chroma_db \
    && chown -R "${APP_UID}:${APP_GID}" /home/app/.cache

# The source list of the corpus download, read-only for the app user.
COPY --link data/sources.toml /app/data/sources.toml
COPY --link src /app/src
COPY --link .streamlit/config.toml /app/.streamlit/config.toml

# The project itself, in editable mode (the venv points at /app/src): its
# metadata, the .pth file and the agentic-rag console script. Bytecode
# compilation stays off: it would recompile all of site-packages into this
# layer. --refresh-package rebuilds the metadata every time, because uv's cache
# does not notice edits of README.md or LICENSE, which the metadata contains.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=LICENSE,target=LICENSE \
    UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
        uv sync --locked --no-dev --refresh-package agentic-rag-chatbot-poc

USER ${APP_UID}:${APP_GID}

EXPOSE 8501

# Slim images have no curl; urlopen raises (exit code 1) unless the server
# answers with a success status. The long start period covers the first start,
# which downloads the corpus and embeds it before the UI listens (minutes on the
# CPU); the 2 s start interval reports the container healthy as soon as the UI
# answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]

# Prepares the knowledge base when INGEST_ON_START is on, then replaces itself
# with Streamlit, which so becomes the main process and receives the signals.
CMD ["agentic-rag", "serve", "--address", "0.0.0.0", "--port", "8501"]
