"""Functional evaluation of the agentic workflow on a small question set (plan section 5.6).

The evaluation reads the question set (``data/eval/questions.jsonl``: 17 questions with
reference answers, expected documents and expected intents), runs every question through the
full main graph or through one of the nodes an item can drive on its own, scores the outcomes
and writes a JSON report and its Markdown summary to ``data/eval/results/``;
``docs/evaluation.md`` discusses the committed runs. The ``agentic-rag eval`` command is the
entry point.

Modules:

- ``dataset``: the ``EvalItem`` record, the JSON Lines loader and the default dataset path.
- ``metrics``: retrieval hit@k and routing accuracy (pure functions); answer correctness and
  faithfulness, judged by a local LLM.
- ``runner``: ``run_evaluation``, the evaluable nodes (``NODE_TARGETS``), the report models,
  which are the format of the committed result files, and their Markdown summary.

The reports share their timestamp and settings snapshot with the load-test report through
``agentic_rag.reports``. The questions are about the frontend documentation corpus (plan
decision 8).
"""
