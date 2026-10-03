"""Load test of the compiled main graph (plan section 5.6).

A harness sends N queries (50-200) directly to the compiled main graph, without an HTTP layer
in between (plan decision 10): a pool of ``concurrency`` threads calls ``graph.invoke``, after
a few warm-up queries that are reported separately. It records the end-to-end latency of every
request and the per-node latency from the request's trace, and reports mean / p50 / p95 / p99
/ max latency, throughput and error rate. Running it once with ``LLM_PROVIDER=fake`` and once
with ``ollama`` isolates the LLM's share of the latency, the core of the bottleneck analysis
in ``docs/performance.md``. The ``agentic-rag loadtest`` command is the entry point.

Modules:

- ``runner``: the latency statistics, the report model, which is the format of the committed
  result files, ``run_load_test``, whose docstring explains the thread pool, and the
  report writer and summary.
"""
