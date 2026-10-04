"""Regression coverage for retrieval, verification, evaluation and source presentation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agentic_rag.agent import graph, nodes, routing
from agentic_rag.agent.state import Subtask, SubtaskResult
from agentic_rag.config import Settings
from agentic_rag.evaluation import runner
from agentic_rag.evaluation.dataset import EvalItem, EvalMessage, load_dataset
from agentic_rag.evaluation.metrics import complete_evidence_at_k
from agentic_rag.ingestion.chunking import split_documents
from agentic_rag.llm import FakeRule, ScriptedChatModel
from agentic_rag.rag import nodes as rag_nodes
from agentic_rag.rag.lexical import KeywordSearch, chunk_key, fuse_rankings
from agentic_rag.rag.state import Source
from agentic_rag.ui.components import link_citations


def test_unreadable_verification_withholds_draft_without_looping():
    model = ScriptedChatModel(default_reply="not valid JSON")
    state = {
        "messages": [HumanMessage("question")],
        "intent": "single",
        "draft_answer": "An unsupported claim",
        "subtask_results": [],
    }
    update = nodes.verify_answer(state, chat_model=model)
    state.update(update)
    assert state["verdict"] == "unavailable"
    assert routing.route_after_verify(state, max_retries=2) == "finalize_response"
    answer = nodes.finalize_response(state)["answer"]
    assert "unsupported claim" not in answer
    assert "couldn't verify" in answer


def test_exact_tool_path_makes_only_one_llm_call(settings: Settings, monkeypatch):
    model = ScriptedChatModel(
        rules=[
            FakeRule(
                pattern="^You are the router",
                reply='{"intent":"tool","question":"Contrast of black on white",'
                '"language":"English","tool_name":"check_contrast",'
                '"tool_args":{"foreground":"#000","background":"#fff"}}',
            )
        ]
    )
    monkeypatch.setattr(graph, "get_chat_model", lambda settings: model)
    output = graph.build_agent_graph(settings).invoke(
        {"messages": [HumanMessage("Black on white")]}
    )
    assert len(model.prompts) == 1
    assert "21.00:1" in output["answer"] or "21:1" in output["answer"]
    assert "verify_answer" not in [event.node for event in output["trace"]]
    assert "source [1]" not in output["answer"]


@pytest.mark.parametrize("kind", ["mixed", "failed", "unknown", "multiple"])
def test_fast_path_does_not_accept_mixed_failed_or_unknown_tools(kind):
    task = Subtask(id="s1", kind="tool", input="contrast", tool_name="check_contrast")
    result = SubtaskResult(subtask_id="s1", kind="tool", output="21:1", tool_name="check_contrast")
    state = {"intent": "tool", "subtasks": [task], "subtask_results": [result]}
    if kind == "mixed":
        state["intent"] = "complex"
    elif kind == "failed":
        state["subtask_results"] = [result.model_copy(update={"ok": False, "error": "failed"})]
    elif kind == "unknown":
        state["subtasks"] = [task.model_copy(update={"tool_name": "custom"})]
        state["subtask_results"] = [result.model_copy(update={"tool_name": "custom"})]
    else:
        state["subtask_results"] = [result, result]
    assert routing.route_after_synthesize(state) == "verify_answer"


class CorpusStore:
    """Minimal paged Chroma stand-in for testing FTS construction."""

    def __init__(self, documents):
        self.documents = documents
        self.reads = 0

    def get(self, *, limit, offset, include):
        self.reads += 1
        docs = self.documents[offset : offset + limit]
        return {
            "ids": [chunk_key(d) for d in docs],
            "documents": [d.page_content for d in docs],
            "metadatas": [d.metadata for d in docs],
        }


def doc(identifier, text):
    return Document(page_content=text, metadata={"chunk_id": identifier, "source": identifier})


def test_keyword_index_keeps_api_tokens_and_builds_once_across_threads():
    store = CorpusStore(
        [
            doc("selector", ":has() selects a parent"),
            doc("prose", "This page has examples"),
            doc("nuxt", "useState is a shared ref"),
        ]
    )
    search = KeywordSearch(lambda: store)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: search(":has()", 4), range(8)))
    assert all([chunk_key(d) for d in result] == ["selector"] for result in results)
    assert store.reads == 2  # One populated page and one terminal empty page.
    assert [chunk_key(d) for d in search('useState " OR NOT *', 4)] == ["nuxt"]
    assert search("the and", 4) == []


def test_fusion_keeps_keyword_only_evidence_and_never_fakes_cosine_scores():
    semantic = [(doc("shared", "computed"), 0.91), (doc("other", "other"), 0.88)]
    lexical = [doc("missing", "computed getter"), semantic[0][0]]
    ranked = fuse_rankings(semantic, lexical, 3)
    assert chunk_key(ranked[0][0]) == "shared"
    assert next(score for d, score in ranked if chunk_key(d) == "missing") == -1.0


def test_grading_can_recover_a_candidate_beyond_four_and_caps_final_context():
    documents = [doc(str(i), f"Candidate {i}") for i in range(20)]
    state = {"query": "computed getter", "documents": documents, "scores": [0.9] * 20}
    model = ScriptedChatModel(default_reply='{"relevant": [12, 13, 14, 15, 16]}')
    graded = rag_nodes.grade_documents(state, min_score=0.83, chat_model=model, top_k=4)
    assert [chunk_key(d) for d in graded["documents"]] == ["11", "12", "13", "14"]
    assert len(model.prompts) == 1


def test_keyword_only_match_reaches_grader_and_has_no_fake_similarity():
    state = {
        "query": "useState",
        "documents": [doc("nuxt", "Nuxt useState")],
        "scores": [-1.0],
        "keyword_matches": ["nuxt"],
    }
    graded = rag_nodes.grade_documents(
        state,
        min_score=0.83,
        chat_model=ScriptedChatModel(default_reply='{"relevant": [1]}'),
        top_k=4,
    )
    context = rag_nodes.build_context(graded)
    assert context["sources"][0].score is None


def test_complete_evidence_requires_both_sides_but_allows_alternative_pages():
    groups = [["react"], ["nuxt-guide", "nuxt-api"]]
    assert complete_evidence_at_k([["react"]], groups, 4) is False
    assert complete_evidence_at_k([["react"], ["nuxt-api"]], groups, 4) is True
    assert complete_evidence_at_k([["react", "noise", "nuxt-guide"]], groups, 2) is False
    assert complete_evidence_at_k([], [], 4) is None
    with pytest.raises(ValueError):
        complete_evidence_at_k([], [[]], 4)


def test_follow_up_history_is_forwarded_and_complete_evidence_is_scored():
    item = EvalItem(
        id="follow",
        question="And Nuxt?",
        reference_answer="A ref",
        history=[EvalMessage(role="user", content="Compare with React useState")],
        expected_document_groups=[["react"], ["nuxt"]],
    )
    seen = []

    def run(question, *, history):
        seen.extend(history)
        return runner._Outcome(answer="A ref", retrieved=[["react"]])

    result = runner._evaluate(item, run, None, target="graph", node=None, k=4)
    assert seen == item.history
    assert result.complete_evidence_at_k is False


def test_citations_use_markdown_references_without_rewriting_code_or_unsafe_urls():
    sources = [
        Source(chunk_id="one", source="doc", content="text", url="https://example.org/docs(a)"),
        Source(chunk_id="two", source="local", content="local", url="javascript:alert(1)"),
    ]
    original = "A claim [1].\n\n`array[1]`\n\n```js\narray[1]\n```"
    linked = link_citations(original, sources)
    assert linked.startswith(original)
    assert "[1]: <https://example.org/docs%28a%29>" in linked
    assert "javascript:" not in linked
    assert "[2]:" not in linked


def test_snapshot_changes_chunk_identity_and_survives_context_building():
    document = doc("original", "The documentation")
    document.metadata["revision"] = "a" * 40
    first = split_documents([document])[0]
    document.metadata["revision"] = "b" * 40
    second = split_documents([document])[0]
    assert first.metadata["chunk_id"] != second.metadata["chunk_id"]
    context = rag_nodes.build_context({"documents": [first], "scores": [0.9]})
    assert context["sources"][0].revision == "a" * 40


def test_holdout_is_separate_valid_and_covers_followups_comparisons_and_boundaries():
    baseline = load_dataset(Path("data/eval/questions.jsonl"))
    holdout = load_dataset(Path("data/eval/holdout.jsonl"))
    assert len(holdout) >= 24
    assert {q.question for q in baseline}.isdisjoint(q.question for q in holdout)
    assert any(q.history for q in holdout)
    assert any(len(q.expected_document_groups) > 1 for q in holdout)
    assert {"hungarian", "version-boundary", "out-of-scope"} <= {
        tag for q in holdout for tag in q.tags
    }
