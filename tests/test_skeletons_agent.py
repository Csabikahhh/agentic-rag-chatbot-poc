"""Tests for the skeleton of the main agentic workflow: nodes, routing, tools and graph.

The skeleton fixes the node names, the signatures and the documented topology of plan section
5.1; the behaviour arrives in Phase 4. These tests guard that contract: the names match the
plan, every node is a traced function with the right state type and explicit dependencies,
the routing annotations name only known nodes (exactly the destinations of the plan's routing
rules), the factories require their settings, and every stub raises the agreed
PlannedFeatureError. That importing the package stays free of heavy libraries is checked in
test_imports.py.
"""

import inspect
import types
import typing
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.types import Send
from pydantic import ValidationError

from agentic_rag.agent import graph, nodes, routing, tools
from agentic_rag.agent.state import AgentState, Subtask, SubtaskInput
from agentic_rag.agent.tools import SEARCH_KNOWLEDGE_BASE, SearchKnowledgeBaseTool
from agentic_rag.cli import main
from agentic_rag.config import Settings
from agentic_rag.errors import PlannedFeatureError, planned
from agentic_rag.llm import ScriptedChatModel
from agentic_rag.rag.state import RagInput, RagOutput

PLAN_NODE_NAMES = (
    "analyze_request",
    "plan_subtasks",
    "run_rag_subtask",
    "call_tool",
    "synthesize_answer",
    "verify_answer",
    "finalize_response",
)
SEND_WORKERS = frozenset({"run_rag_subtask", "call_tool"})

# The routing rules of plan section 5.1: (nodes returned by name, nodes reached through Send).
PLANNED_ROUTES: dict[str, tuple[set[str], set[str]]] = {
    "route_after_analyze": (
        {"finalize_response", "plan_subtasks"},
        {"run_rag_subtask", "call_tool"},
    ),
    "dispatch_subtasks": (set(), {"run_rag_subtask", "call_tool"}),
    "route_after_verify": ({"plan_subtasks", "finalize_response"}, set()),
}

# What build_agent_graph binds to each node; it mirrors the node kinds of plan section 5.1.
NODE_DEPENDENCIES: dict[str, set[str]] = {
    "analyze_request": {"chat_model", "tools"},
    "plan_subtasks": {"chat_model", "tools"},
    "run_rag_subtask": {"search_tool"},
    "call_tool": {"tools"},
    "synthesize_answer": {"chat_model"},
    "verify_answer": {"chat_model"},
    "finalize_response": set(),
}
DEPENDENCY_TYPES: dict[str, Any] = {
    "chat_model": BaseChatModel,
    "tools": Mapping[str, BaseTool],
    "search_tool": SearchKnowledgeBaseTool,
}

# The factories of the agent package: each takes the settings as its required first argument.
SETTINGS_FACTORIES: list[Callable[..., object]] = [
    graph.build_agent_graph,
    tools.get_tools,
    tools.get_non_retrieval_tools,
]


def _phase_4(qualified_name: str) -> str:
    """Return the agreed message of a Phase 4 stub."""
    return str(planned(qualified_name, 4))


def _empty_rag_output(rag_input: RagInput) -> RagOutput:
    """Stand in for the RAG subgraph: find nothing."""
    return RagOutput(context="", sources=[], trace=[])


def _search_tool() -> SearchKnowledgeBaseTool:
    """Build the search tool over the stand-in RAG subgraph."""
    return SearchKnowledgeBaseTool(rag_graph=RunnableLambda(_empty_rag_output))


def _dependencies() -> dict[str, Any]:
    """Return stand-ins for every dependency that build_agent_graph binds to the nodes."""
    return {"chat_model": ScriptedChatModel(), "tools": {}, "search_tool": _search_tool()}


def _state_for(node_name: str) -> AgentState | SubtaskInput:
    """Return a minimal input of the node's state type."""
    if node_name == "run_rag_subtask":
        subtask = Subtask(id="s1", kind="retrieve", input="alpha")
    elif node_name == "call_tool":
        subtask = Subtask(id="s1", kind="tool", input="compute alpha", tool_name="some_tool")
    else:
        return AgentState(messages=[HumanMessage(content="What is alpha?")])
    return SubtaskInput(subtask=subtask, question="What is alpha?")


def _union_members(annotation: Any) -> tuple[Any, ...]:
    """Return the members of a union annotation, or the annotation alone."""
    if get_origin(annotation) in (typing.Union, types.UnionType):
        return get_args(annotation)
    return (annotation,)


def _literal_names(annotation: Any) -> set[str]:
    """Return the string values of a Literal annotation; empty for anything else."""
    if get_origin(annotation) is not Literal:
        return set()
    values = get_args(annotation)
    assert all(isinstance(value, str) for value in values)
    return set(values)


def _route_targets(route: Callable[..., Any]) -> tuple[set[str], set[str]]:
    """Split a routing function's return annotation into named nodes and Send targets."""
    annotation = get_type_hints(route, include_extras=True)["return"]
    named: set[str] = set()
    sent: set[str] = set()
    for member in _union_members(annotation):
        if get_origin(member) is Annotated:
            base, *metadata = get_args(member)
            assert base == list[Send], f"{route.__name__}: only list[Send] may carry targets"
            for item in metadata:
                sent |= _literal_names(item)
        else:
            assert member != list[Send], f"{route.__name__}: annotate fan-outs as SubtaskSends"
            names = _literal_names(member)
            assert names, f"{route.__name__}: unexpected return annotation member {member!r}"
            named |= names
    return named, sent


# --- nodes -----------------------------------------------------------------------------------


def test_node_names_match_the_plan() -> None:
    """NODE_NAMES is the plan's node table in order: at least five distinct nodes."""
    assert graph.NODE_NAMES == PLAN_NODE_NAMES
    assert len(graph.NODE_NAMES) >= 5
    assert len(set(graph.NODE_NAMES)) == len(graph.NODE_NAMES)
    assert set(NODE_DEPENDENCIES) == set(graph.NODE_NAMES)


@pytest.mark.parametrize("name", PLAN_NODE_NAMES)
def test_every_node_is_a_traced_function_named_after_its_node(name: str) -> None:
    """Each name has a sync function in nodes.py, wrapped by traced and keeping its name."""
    node = getattr(nodes, name)
    assert name in nodes.__all__
    assert inspect.isfunction(node)
    assert not inspect.iscoroutinefunction(node), "nodes are sync (the UI streams with stream)"
    assert node.__name__ == name
    original = inspect.unwrap(node)
    assert original is not node, "decorate the node with agentic_rag.tracing.traced"
    assert original.__name__ == name


@pytest.mark.parametrize("name", PLAN_NODE_NAMES)
def test_nodes_take_their_state_and_explicit_dependencies(name: str) -> None:
    """The state is the only positional parameter; dependencies are keyword-only and bound."""
    node = getattr(nodes, name)
    hints = get_type_hints(node)
    state_param, *dependency_params = inspect.signature(node).parameters.values()
    assert state_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert hints[state_param.name] is (SubtaskInput if name in SEND_WORKERS else AgentState)
    assert hints["return"] == dict[str, Any]
    assert {param.name for param in dependency_params} == NODE_DEPENDENCIES[name]
    for param in dependency_params:
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty, "dependencies are bound, not defaulted"
        assert hints[param.name] == DEPENDENCY_TYPES[param.name]


def test_send_targets_are_exactly_the_nodes_that_take_a_subtask_input() -> None:
    """The Send workers of the routing module are the nodes whose input is a SubtaskInput."""
    workers = {
        name
        for name in graph.NODE_NAMES
        if next(iter(get_type_hints(getattr(nodes, name)).values())) is SubtaskInput
    }
    assert workers == set(get_args(routing.SubtaskWorker)) == SEND_WORKERS


@pytest.mark.parametrize("name", PLAN_NODE_NAMES)
def test_node_stubs_raise_planned_feature_error_for_phase_4(name: str) -> None:
    """Every node stub raises the agreed error and message, through the traced wrapper."""
    dependencies = _dependencies()
    kwargs = {param: dependencies[param] for param in NODE_DEPENDENCIES[name]}
    with pytest.raises(PlannedFeatureError) as excinfo:
        getattr(nodes, name)(_state_for(name), **kwargs)
    assert str(excinfo.value) == _phase_4(f"agentic_rag.agent.nodes.{name}")


# --- routing ---------------------------------------------------------------------------------


def test_routing_annotations_mention_only_known_node_names() -> None:
    """Every routing function's return annotation names only nodes of NODE_NAMES."""
    routes = [
        getattr(routing, name)
        for name in routing.__all__
        if inspect.isfunction(getattr(routing, name))
    ]
    assert {route.__name__ for route in routes} == set(PLANNED_ROUTES)
    for route in routes:
        named, sent = _route_targets(route)
        assert named | sent <= set(graph.NODE_NAMES), route.__name__


@pytest.mark.parametrize("name", list(PLANNED_ROUTES))
def test_routing_annotations_list_the_planned_destinations(name: str) -> None:
    """Each routing point reaches exactly the destinations of the plan's routing rules."""
    assert _route_targets(getattr(routing, name)) == PLANNED_ROUTES[name]


@pytest.mark.parametrize("name", list(PLANNED_ROUTES))
def test_routing_functions_read_the_agent_state(name: str) -> None:
    """Routes take the AgentState; only the verify loop gets a bound max_retries."""
    route = getattr(routing, name)
    hints = get_type_hints(route)
    state_param, *bound_params = inspect.signature(route).parameters.values()
    assert hints[state_param.name] is AgentState
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in bound_params)
    expected = {"max_retries": int} if name == "route_after_verify" else {}
    assert {param.name: hints[param.name] for param in bound_params} == expected


@pytest.mark.parametrize("name", list(PLANNED_ROUTES))
def test_routing_stubs_raise_planned_feature_error_for_phase_4(name: str) -> None:
    """Every routing stub raises the agreed error and message."""
    kwargs = {"max_retries": 2} if name == "route_after_verify" else {}
    with pytest.raises(PlannedFeatureError) as excinfo:
        getattr(routing, name)(_state_for("verify_answer"), **kwargs)
    assert str(excinfo.value) == _phase_4(f"agentic_rag.agent.routing.{name}")


# --- tools and graph -------------------------------------------------------------------------


def test_search_tool_has_its_final_name_schema_and_response_format() -> None:
    """The retrieval tool's interface is final: name, one-field schema, content and artifact."""
    tool = _search_tool()
    assert tool.name == SEARCH_KNOWLEDGE_BASE == "search_knowledge_base"
    assert tool.description.strip()
    assert tool.response_format == "content_and_artifact"
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]
    assert "rag_graph" not in tool.model_dump()


def test_search_tool_validates_its_query_and_its_subgraph() -> None:
    """An empty query and a subgraph that is not a runnable are rejected."""
    with pytest.raises(ValidationError):
        _search_tool().invoke({"query": ""})
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseTool.model_validate({"rag_graph": "not a runnable"})


@pytest.mark.parametrize("factory", SETTINGS_FACTORIES, ids=lambda factory: factory.__name__)
def test_factories_require_their_settings(factory: Callable[..., object]) -> None:
    """The settings are the required first argument: no default and no global fallback."""
    settings_param = next(iter(inspect.signature(factory).parameters.values()))
    assert settings_param.name == "settings"
    assert settings_param.default is inspect.Parameter.empty
    assert get_type_hints(factory)["settings"] is Settings
    with pytest.raises(TypeError):
        factory()


STUB_CALLS: list[tuple[str, Callable[[Settings], object]]] = [
    ("agentic_rag.agent.graph.build_agent_graph", graph.build_agent_graph),
    (
        "agentic_rag.agent.tools.get_tools",
        lambda settings: tools.get_tools(settings, rag_graph=RunnableLambda(_empty_rag_output)),
    ),
    ("agentic_rag.agent.tools.get_non_retrieval_tools", tools.get_non_retrieval_tools),
    (
        "agentic_rag.agent.tools.SearchKnowledgeBaseTool._run",
        lambda settings: _search_tool().invoke({"query": "alpha"}),
    ),
]


@pytest.mark.parametrize(
    ("qualified_name", "call"), STUB_CALLS, ids=[name for name, _ in STUB_CALLS]
)
def test_graph_and_tool_stubs_raise_planned_feature_error_for_phase_4(
    qualified_name: str, call: Callable[[Settings], object], settings: Settings
) -> None:
    """The graph builder, the tool factories and the search tool raise the agreed error."""
    with pytest.raises(PlannedFeatureError) as excinfo:
        call(settings)
    assert str(excinfo.value) == _phase_4(qualified_name)


def test_cli_export_graph_reports_the_planned_phase(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI turns the builder's PlannedFeatureError into its message and exit code 1."""
    assert main(["export-graph", "--graph", "agent"]) == 1
    assert _phase_4("agentic_rag.agent.graph.build_agent_graph") in capsys.readouterr().err
