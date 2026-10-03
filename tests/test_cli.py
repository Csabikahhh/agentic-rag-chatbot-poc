"""Tests for the command-line interface: agentic_rag.cli and ``python -m agentic_rag``.

The handlers import their target modules lazily. These tests put stand-in modules into
``sys.modules``, so they do not depend on the skeletons and implementations of other modules.
"""

import json
import logging
import signal
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel, ValidationError

from agentic_rag import __version__, cli
from agentic_rag.config import Settings
from agentic_rag.errors import ConfigurationError, InvalidArgumentError, planned

COMMANDS = ("ingest", "eval", "loadtest", "export-graph", "config", "serve")


def install_module(
    monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any
) -> types.ModuleType:
    """Make ``import name`` return a stand-in module with the given attributes for one test."""
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def tiny_graph(node_name: str) -> Any:
    """Compile a one-node LangGraph graph, a stand-in for the real graph builders."""
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def step(state: State) -> dict[str, int]:
        return {"value": state["value"] + 1}

    builder = StateGraph(State)
    builder.add_node(node_name, step)
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()


# --- help, version and usage errors -------------------------------------------------------


def test_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0

    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0

    assert f"usage: agentic-rag {command}" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0

    assert capsys.readouterr().out.strip() == f"agentic-rag {__version__}"


@pytest.mark.parametrize("argv", [[], ["unknown-command"]])
def test_missing_or_unknown_command_is_a_usage_error(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2

    assert "usage: agentic-rag" in capsys.readouterr().err


def test_build_parser_registers_the_commands() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["loadtest"])

    assert (args.command, args.requests, args.concurrency, args.warmup) == ("loadtest", 100, 4, 3)
    assert args.output_dir is None


# --- config -------------------------------------------------------------------------------


def test_config_prints_the_effective_settings(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["config"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{name}={value}" for name, value in settings.as_env().items()]
    assert "LLM_PROVIDER=fake" in lines
    assert f"DATA_DIR={settings.data_dir.as_posix()}" in lines


def test_config_reflects_environment_overrides(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("INGEST_ON_START", "false")

    assert cli.main(["config"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "TOP_K=9" in lines
    assert "INGEST_ON_START=false" in lines


def test_invalid_configuration_is_reported_by_variable_name(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert cli.main(["config"]) == 2

    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert lines[0].startswith("Invalid configuration")
    # One line per variable: its name, pydantic's message and the rejected value.
    assert [line.split(":")[0].strip() for line in lines[1:]] == ["LLM_PROVIDER", "TOP_K"]
    assert lines[1].endswith("(got 'openai')")
    assert lines[2].endswith("(got '0')")
    assert captured.out == ""


def test_logging_follows_the_log_level_setting(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    assert cli.main(["config"]) == 0

    assert logging.getLogger().level == logging.DEBUG


# --- errors of the commands ---------------------------------------------------------------

STUBS = [
    (["ingest"], "agentic_rag.ingestion.index", "build_index", 2),
    (["eval"], "agentic_rag.evaluation.runner", "run_evaluation", 7),
    (["loadtest"], "agentic_rag.loadtest.runner", "run_load_test", 8),
    (["export-graph", "--graph", "agent"], "agentic_rag.agent.graph", "build_agent_graph", 4),
    (["export-graph", "--graph", "rag"], "agentic_rag.rag.graph", "build_rag_graph", 3),
]


def install_runner(monkeypatch: pytest.MonkeyPatch, run_evaluation: Callable[..., Any]) -> None:
    """Replace the evaluation runner module with one whose run_evaluation is given."""
    install_module(
        monkeypatch,
        "agentic_rag.evaluation.runner",
        run_evaluation=run_evaluation,
        render_summary=str,
    )


def raising(error: BaseException) -> Callable[..., Any]:
    """A stand-in command target that raises ``error``."""

    def run(*args: Any, **kwargs: Any) -> Any:
        raise error

    return run


@pytest.mark.parametrize(("argv", "module", "function", "phase"), STUBS)
def test_planned_feature_prints_its_message_and_exits_with_1(
    argv: list[str],
    module: str,
    function: str,
    phase: int,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = planned(f"{module}.{function}", phase)
    # render_summary is what eval prints; the other modules ignore the extra attribute.
    install_module(monkeypatch, module, **{function: raising(error)}, render_summary=str)

    assert cli.main(argv) == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == str(error)
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "error",
    [
        # A library's own NotImplementedError is a real failure, not a planned stub.
        pytest.param(
            NotImplementedError("StructuredTool does not support sync invocation."),
            id="library-not-implemented",
        ),
        pytest.param(NotImplementedError(), id="bare-not-implemented"),
        pytest.param(RuntimeError("index is corrupt"), id="runtime-error"),
    ],
)
def test_other_errors_of_a_command_propagate_with_their_traceback(
    error: Exception, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_runner(monkeypatch, raising(error))

    with pytest.raises(type(error)) as excinfo:
        cli.main(["eval"])

    assert excinfo.value is error


def test_a_validation_error_inside_a_command_is_not_a_configuration_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_evaluation(settings: Settings, **kwargs: Any) -> Any:
        return Settings.model_validate({**settings.model_dump(), "top_k": 0})

    install_runner(monkeypatch, run_evaluation)

    # Only the settings loaded at start-up are configuration; this one is a bug in the command.
    with pytest.raises(ValidationError):
        cli.main(["eval"])


def test_invalid_argument_error_is_reported_as_a_usage_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    message = "node 'verify_answer' cannot run from an evaluation item; choose analyze_request"
    error = InvalidArgumentError(message)
    install_runner(monkeypatch, raising(error))

    assert cli.main(["eval", "--target", "node", "--node", "verify_answer"]) == 2

    captured = capsys.readouterr()
    assert captured.err.startswith("usage: agentic-rag eval")
    assert captured.err.rstrip().endswith(f"agentic-rag eval: error: {message}")
    assert captured.out == ""


def test_configuration_error_of_a_command_exits_with_2(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = ConfigurationError("the index was built with another embedding model")
    install_module(monkeypatch, "agentic_rag.ingestion.index", build_index=raising(error))

    assert cli.main(["ingest"]) == 2

    captured = capsys.readouterr()
    assert captured.err.strip() == f"Invalid configuration: {error}"
    assert captured.out == ""


# --- ingest, eval and loadtest forward their options --------------------------------------


class Report(BaseModel):
    """A stand-in result model; the CLI prints models as JSON."""

    name: str
    count: int


def recorder(calls: list[dict[str, Any]], result: Any) -> Callable[..., Any]:
    """A stand-in runner that records how it was called."""

    def run(settings: Settings, **kwargs: Any) -> Any:
        calls.append({"settings": settings, **kwargs})
        return result

    return run


def test_ingest_passes_rebuild_and_prints_the_stats(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []
    stats = Report(name="index", count=5)
    install_module(monkeypatch, "agentic_rag.ingestion.index", build_index=recorder(calls, stats))

    assert cli.main(["ingest"]) == 0
    assert json.loads(capsys.readouterr().out) == {"name": "index", "count": 5}
    assert cli.main(["ingest", "--rebuild"]) == 0

    assert calls == [
        {"settings": settings, "rebuild": False},
        {"settings": settings, "rebuild": True},
    ]


class StandInDownloadError(RuntimeError):
    """Stands in for agentic_rag.ingestion.download.DownloadError."""


def test_ingest_download_fetches_the_sources_before_building_the_index(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    order: list[str] = []
    sources = tmp_path / "sources.toml"
    sources.write_text("", encoding="utf-8")

    def download_sources(settings: Settings, *, sources_file: Path) -> None:
        order.append(f"download {sources_file}")

    def build_index(settings: Settings, *, rebuild: bool) -> Report:
        order.append(f"build rebuild={rebuild}")
        return Report(name="index", count=3)

    install_module(
        monkeypatch,
        "agentic_rag.ingestion.download",
        download_sources=download_sources,
        DownloadError=StandInDownloadError,
    )
    install_module(monkeypatch, "agentic_rag.ingestion.index", build_index=build_index)

    assert cli.main(["ingest", "--download", "--sources", str(sources), "--rebuild"]) == 0

    assert order == [f"download {sources}", "build rebuild=True"]
    assert json.loads(capsys.readouterr().out) == {"name": "index", "count": 3}


def test_ingest_download_uses_the_repository_source_list_by_default(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    install_module(
        monkeypatch,
        "agentic_rag.ingestion.download",
        download_sources=lambda settings, *, sources_file: calls.append(sources_file),
        DownloadError=StandInDownloadError,
    )
    install_module(
        monkeypatch,
        "agentic_rag.ingestion.index",
        build_index=lambda settings, *, rebuild: Report(name="index", count=0),
    )

    # The tests run from the repository root, where data/sources.toml exists.
    assert cli.main(["ingest", "--download"]) == 0

    assert calls == [Path("data/sources.toml")]


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (
            ["ingest", "--sources", "data/sources.toml"],
            "--sources can only be used with --download",
        ),
        (["ingest", "--download", "--sources", "missing.toml"], "file not found: missing.toml"),
    ],
)
def test_ingest_usage_errors(
    argv: list[str], error: str, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2

    err = capsys.readouterr().err
    assert "usage: agentic-rag ingest" in err
    assert error in err


def test_a_failed_download_prints_its_message_and_exits_with_1(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sources = tmp_path / "sources.toml"
    sources.write_text("", encoding="utf-8")
    error = StandInDownloadError("git is required to download the corpus")
    install_module(
        monkeypatch,
        "agentic_rag.ingestion.download",
        download_sources=raising(error),
        DownloadError=StandInDownloadError,
    )

    assert cli.main(["ingest", "--download", "--sources", str(sources)]) == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == f"agentic-rag ingest: {error}"
    assert captured.out == ""


def test_a_missing_corpus_prints_its_message_and_exits_with_1(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = FileNotFoundError("The corpus in data/raw has no documents")
    install_module(monkeypatch, "agentic_rag.ingestion.index", build_index=raising(error))

    assert cli.main(["ingest"]) == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == f"agentic-rag ingest: {error}"
    assert "Traceback" not in captured.err


def test_ingest_builds_a_real_index_from_the_corpus(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (settings.data_dir / "guide.md").write_text("## Setup\n\nInstall it.\n", encoding="utf-8")

    assert cli.main(["ingest"]) == 0

    stats = json.loads(capsys.readouterr().out)
    assert (stats["files"], stats["documents"], stats["chunks"]) == (1, 1, 1)
    assert stats["embedding_provider"] == "fake"


def test_eval_forwards_its_options_and_prints_the_summary(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"question": "?"}\n', encoding="utf-8")
    calls: list[dict[str, Any]] = []
    report = Report(name="eval", count=1)
    install_module(
        monkeypatch,
        "agentic_rag.evaluation.runner",
        run_evaluation=recorder(calls, report),
        render_summary=lambda report: f"# Summary of {report.name}\n",
    )

    assert cli.main(["eval", "--judge-model", "judge:7b"]) == 0
    assert capsys.readouterr().out == "# Summary of eval\n"
    argv = ["eval", "--target", "node", "--node", "analyze_request"]
    argv += ["--dataset", str(dataset), "--output-dir", str(tmp_path / "out")]
    assert cli.main(argv) == 0

    assert calls == [
        {
            "settings": settings,
            "target": "graph",
            "node": None,
            "dataset_path": None,
            "output_dir": None,
            "judge_model": "judge:7b",
        },
        {
            "settings": settings,
            "target": "node",
            "node": "analyze_request",
            "dataset_path": dataset,
            "output_dir": tmp_path / "out",
            "judge_model": None,
        },
    ]


def test_eval_without_an_index_prints_its_message_and_exits_with_1(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    # The committed question set, in fake mode, but no index was built.
    assert cli.main(["eval"]) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("agentic-rag eval: The vector index 'documents' does not exist")
    assert "Traceback" not in captured.err


def test_eval_reports_an_invalid_question_set_with_its_line(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id": "q1"}\n', encoding="utf-8")

    assert cli.main(["eval", "--dataset", str(dataset)]) == 1

    assert capsys.readouterr().err.startswith(f"agentic-rag eval: {dataset}:1: invalid item")


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["eval", "--target", "node"], "--node is required"),
        (["eval", "--node", "analyze_request"], "only be used with --target node"),
        (["eval", "--dataset", "missing.jsonl"], "file not found"),
        (["eval", "--target", "everything"], "invalid choice"),
        (
            ["eval", "--target", "node", "--node", "analyze_request", "--judge-model", "x"],
            "--judge-model can only be used with --target graph",
        ),
    ],
)
def test_eval_usage_errors(
    argv: list[str], error: str, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2

    err = capsys.readouterr().err
    assert "usage: agentic-rag eval" in err
    assert error in err


def test_loadtest_forwards_its_options_and_prints_the_summary(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []
    install_module(
        monkeypatch,
        "agentic_rag.loadtest.runner",
        run_load_test=recorder(calls, Report(name="load", count=100)),
        render_summary=lambda report: f"# Load test {report.name}\n",
    )
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("", encoding="utf-8")

    assert cli.main(["loadtest"]) == 0
    assert capsys.readouterr().out == "# Load test load\n"
    argv = ["loadtest", "--requests", "50", "--concurrency", "8", "--warmup", "0"]
    assert cli.main([*argv, "--output-dir", str(tmp_path), "--dataset", str(dataset)]) == 0

    assert calls == [
        {
            "settings": settings,
            "requests": 100,
            "concurrency": 4,
            "warmup": 3,
            "output_dir": None,
            "dataset_path": None,
        },
        {
            "settings": settings,
            "requests": 50,
            "concurrency": 8,
            "warmup": 0,
            "output_dir": tmp_path,
            "dataset_path": dataset,
        },
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["loadtest", "--requests", "0"],
        ["loadtest", "--requests", "ten"],
        ["loadtest", "--requests", "1.5"],
        ["loadtest", "--concurrency", "0"],
        ["loadtest", "--concurrency", "-2"],
        ["loadtest", "--warmup", "-1"],
    ],
)
def test_loadtest_rejects_invalid_numbers(
    argv: list[str], settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2

    err = capsys.readouterr().err
    assert f"argument {argv[1]}" in err


# --- export-graph -------------------------------------------------------------------------


def test_export_graph_prints_both_diagrams_as_markdown(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_module(
        monkeypatch,
        "agentic_rag.agent.graph",
        build_agent_graph=lambda settings: tiny_graph("analyze_request"),
    )
    install_module(
        monkeypatch,
        "agentic_rag.rag.graph",
        build_rag_graph=lambda settings: tiny_graph("retrieve"),
    )

    assert cli.main(["export-graph"]) == 0

    out = capsys.readouterr().out
    assert out.count("```mermaid") == 2
    assert out.index("## Main agentic workflow") < out.index("## RAG subgraph")
    assert out.index("analyze_request") < out.index("retrieve")
    assert "graph TD" in out


def test_export_graph_writes_one_raw_diagram_to_a_file(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_module(
        monkeypatch,
        "agentic_rag.rag.graph",
        build_rag_graph=lambda settings: tiny_graph("retrieve"),
    )
    target = tmp_path / "docs" / "rag.mmd"

    argv = ["export-graph", "--graph", "rag", "--format", "mermaid", "--output", str(target)]
    assert cli.main(argv) == 0

    text = target.read_text(encoding="utf-8")
    assert "```" not in text
    assert "retrieve" in text
    assert "graph TD" in text
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(target) in captured.err


def test_export_graph_raw_mermaid_needs_a_single_graph(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["export-graph", "--format", "mermaid"]) == 2

    assert "choose --graph agent or --graph rag" in capsys.readouterr().err


# --- serve --------------------------------------------------------------------------------


def serve_stand_ins(
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[Settings], Any] | None = None,
    exit_code: int = 0,
) -> tuple[list[Settings], list[list[str]]]:
    """Replace the start-up preparation and the start of Streamlit for one ``serve`` run.

    Returns:
        The settings of every preparation and the command of every start, in call order.
    """
    prepared: list[Settings] = []
    started: list[list[str]] = []

    def prepare_knowledge_base(settings: Settings) -> None:
        prepared.append(settings)
        if prepare is not None:
            prepare(settings)

    def start(command: list[str]) -> int:
        started.append(command)
        return exit_code

    install_module(
        monkeypatch, "agentic_rag.ingestion.prepare", prepare_knowledge_base=prepare_knowledge_base
    )
    install_module(
        monkeypatch, "agentic_rag.ingestion.download", DownloadError=StandInDownloadError
    )
    monkeypatch.setattr(cli, "_start", start)
    return prepared, started


def test_serve_prepares_the_knowledge_base_then_starts_the_ui(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, started = serve_stand_ins(monkeypatch)

    assert cli.main(["serve", "--address", "0.0.0.0", "--port", "8600"]) == 0

    assert prepared == [settings]
    assert started == [
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(cli.UI_SCRIPT),
            "--server.port=8600",
            "--server.address=0.0.0.0",
        ]
    ]
    assert cli.UI_SCRIPT.is_file()


def test_serve_skips_the_preparation_when_ingest_on_start_is_off(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGEST_ON_START", "false")
    prepared, started = serve_stand_ins(monkeypatch, exit_code=3)

    assert cli.main(["serve"]) == 3  # The exit code of Streamlit.

    assert prepared == []
    assert started[0][-1] == "--server.port=8501"  # Streamlit's default address.


def test_serve_starts_the_ui_when_the_preparation_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, started = serve_stand_ins(
        monkeypatch, prepare=raising(StandInDownloadError("git fetch failed: no network"))
    )

    assert cli.main(["serve"]) == 0

    assert len(started) == 1
    (record,) = [
        record
        for record in caplog.records
        if record.name == "agentic_rag.cli" and record.levelno == logging.ERROR
    ]
    assert record.getMessage().startswith(
        "The knowledge base could not be prepared: git fetch failed: no network."
    )


def test_serve_lets_an_unexpected_preparation_error_propagate(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, started = serve_stand_ins(monkeypatch, prepare=raising(RuntimeError("bug")))

    with pytest.raises(RuntimeError, match="bug"):
        cli.main(["serve"])
    assert started == []


def test_sigterm_interrupts_the_preparation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = signal.getsignal(signal.SIGTERM)
    _, started = serve_stand_ins(
        monkeypatch, prepare=lambda settings: signal.raise_signal(signal.SIGTERM)
    )

    assert cli.main(["serve"]) == 130  # What docker stop sends stops the preparation.

    assert started == []
    assert "Interrupted." in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == before


def test_serve_rejects_an_invalid_port(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["serve", "--port", "70000"]) == 2

    assert "must be at most 65535, got 70000" in capsys.readouterr().err


@pytest.mark.parametrize("replace_process", [True, False])
def test_start_replaces_the_process_or_runs_a_child(
    replace_process: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(cli, "_REPLACE_PROCESS", replace_process)
    monkeypatch.setattr(cli.os, "execv", lambda path, args: calls.append(("execv", args)))
    monkeypatch.setattr(cli.subprocess, "call", lambda args: calls.append(("call", args)) or 7)

    result = cli._start(["python", "-m", "streamlit"])

    if replace_process:
        assert calls == [("execv", ["python", "-m", "streamlit"])]
    else:
        assert (calls, result) == ([("call", ["python", "-m", "streamlit"])], 7)


# --- python -m agentic_rag in a fresh interpreter -----------------------------------------
# What --help and config import is checked in test_imports.py.


def run_python(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the interpreter of this test session in a clean working directory."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
        check=False,
    )


def test_python_m_agentic_rag_help(tmp_path: Path) -> None:
    result = run_python(["-m", "agentic_rag", "--help"], cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    for command in COMMANDS:
        assert command in result.stdout


def test_python_m_agentic_rag_reports_a_dotenv_that_is_not_utf8(tmp_path: Path) -> None:
    # What `echo LLM_PROVIDER=fake > .env` writes in Windows PowerShell 5.1: UTF-16 with a BOM.
    (tmp_path / ".env").write_bytes("LLM_PROVIDER=fake\n".encode("utf-16"))

    result = run_python(["-m", "agentic_rag", "config"], cwd=tmp_path)

    assert result.returncode == 2
    assert result.stderr.startswith("Invalid configuration: ")
    assert ".env is not valid UTF-8" in result.stderr
    assert "Save .env as UTF-8" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("raise_statement", "last_line", "traceback"),
    [
        pytest.param(
            "raise planned('agentic_rag.evaluation.runner.run_evaluation', 7)",
            str(planned("agentic_rag.evaluation.runner.run_evaluation", 7)),
            False,
            id="planned-feature",
        ),
        pytest.param(
            "raise NotImplementedError('StructuredTool does not support sync invocation.')",
            "NotImplementedError: StructuredTool does not support sync invocation.",
            True,
            id="library-not-implemented",
        ),
    ],
)
def test_python_m_agentic_rag_exits_with_1_when_a_command_fails(
    raise_statement: str, last_line: str, traceback: bool, tmp_path: Path
) -> None:
    # The real entry point (agentic_rag.__main__) around a stand-in runner that raises: only
    # a planned feature is reported without its traceback.
    script = (
        "import runpy, sys, types\n"
        "from agentic_rag.errors import planned\n"
        "def run_evaluation(settings, **kwargs):\n"
        f"    {raise_statement}\n"
        "runner = types.ModuleType('agentic_rag.evaluation.runner')\n"
        "runner.run_evaluation = run_evaluation\n"
        "runner.render_summary = str\n"
        "sys.modules[runner.__name__] = runner\n"
        "sys.argv = ['agentic-rag', 'eval']\n"
        "runpy.run_module('agentic_rag', run_name='__main__')\n"
    )

    result = run_python(["-c", script], cwd=tmp_path)

    assert result.returncode == 1
    assert result.stderr.rstrip().splitlines()[-1] == last_line
    assert ("Traceback (most recent call last)" in result.stderr) is traceback
    assert result.stdout == ""
