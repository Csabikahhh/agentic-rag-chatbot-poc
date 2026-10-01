"""Functional evaluation of the agentic workflow on a small question set (plan section 5.6).

The evaluation reads the question set (``data/eval/questions.jsonl``: 10-20 questions with
reference answers, expected documents and expected intents), runs every question through the
full main graph or through one of the nodes an item can drive on its own, scores the outcomes
and writes a JSON report to ``data/eval/results/``; the written summary goes to
``docs/evaluation.md``. The ``agentic-rag eval`` command is the entry point.

Modules:

- ``dataset``: the ``EvalItem`` record, the JSON Lines loader and the default dataset path.
- ``metrics``: retrieval hit@k and routing accuracy (pure functions); answer correctness and
  faithfulness, judged by the local LLM (Phase 7).
- ``runner``: the report models, which are the format of the committed result files, the
  evaluable nodes (``NODE_TARGETS``) and ``run_evaluation`` (Phase 7).

The reports share their timestamp and settings snapshot with the load-test report through
``agentic_rag.reports``. The questions themselves are written in Phase 7, once the domain and
the corpus are chosen (plan decision 8).
"""
