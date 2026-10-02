"""Import cost: what each entry point and package group loads in a fresh interpreter.

The project imports its heavy libraries (models, the vector store, text splitting, the UI)
lazily, inside the functions that use them; the light modules (tracing, the literal types, the
reports, the evaluation records and metrics, the load-test statistics) do not load the graph
runtime either. Each group below runs in its own interpreter, which starts with nothing
imported, so a module that an earlier test imported cannot hide an import. The probe imports
the modules of the group one at a time and attributes every forbidden package that appears to
the module whose import loaded it, then runs the group's ``exercise`` (a few lines that use the
entry point offline) and attributes what that loads to the exercise.

The forbidden sets come from ``conftest.py``. All interpreters start together when the first
test of this file needs them, so the file costs about as much as its slowest group.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.config import Settings
from conftest import GRAPH_RUNTIME_MODULES, heavy_modules

REPORT_PREFIX = "IMPORT-PROBE-REPORT "

# Runs in a fresh interpreter. argv[1] is a JSON object with the keys "modules" (imported in
# order), "forbidden" (top-level package names) and "exercise" (code run after the imports;
# it may set ``result``). Prints one line: REPORT_PREFIX and a JSON report.
_PROBE = """
import contextlib, importlib, io, json, sys
spec = json.loads(sys.argv[1])
forbidden = frozenset(spec["forbidden"])

def loaded():
    return {name.partition(".")[0] for name in sys.modules} & forbidden

seen = loaded()
report = {"at start-up": sorted(seen), "imports": {}, "exercise": [], "result": None}
for name in spec["modules"]:
    importlib.import_module(name)
    now = loaded()
    report["imports"][name] = sorted(now - seen)
    seen = now
if spec["exercise"]:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(spec["exercise"], namespace)
    report["result"] = namespace.get("result")
    report["exercise"] = sorted(loaded() - seen)
print(REPORT_PREFIX + json.dumps(report))
""".replace("REPORT_PREFIX", repr(REPORT_PREFIX))


@dataclass(frozen=True)
class ImportGroup:
    """One fresh interpreter: the modules it imports and what they must not load."""

    name: str
    modules: tuple[str, ...]
    forbidden: frozenset[str]
    exercise: str = ""
    """Code run after the imports, with stdout captured; it may set ``result``."""
    expected_result: Any = None
    env: dict[str, str] = field(default_factory=dict)
    """Variables set for the interpreter on top of the isolated environment."""


GROUPS = (
    ImportGroup(
        name="cli",
        modules=("agentic_rag", "agentic_rag.config", "agentic_rag.cli", "agentic_rag.__main__"),
        forbidden=heavy_modules(),
        # --help and config are the commands that must answer instantly, with the default
        # (Ollama and Hugging Face) configuration.
        exercise=(
            "from agentic_rag.cli import main\n"
            "result = [main(['--help']), main(['export-graph', '--help']), main(['config'])]\n"
        ),
        expected_result=[0, 0, 0],
    ),
    ImportGroup(
        name="agent",
        modules=(
            "agentic_rag.agent",
            "agentic_rag.agent.types",
            "agentic_rag.agent.state",
            "agentic_rag.agent.routing",
            "agentic_rag.agent.tools",
            "agentic_rag.agent.nodes",
            "agentic_rag.agent.graph",
        ),
        forbidden=heavy_modules(),
    ),
    ImportGroup(
        name="rag",
        modules=(
            "agentic_rag.embeddings",
            "agentic_rag.ingestion",
            "agentic_rag.ingestion.sources",
            "agentic_rag.ingestion.markdown",
            "agentic_rag.ingestion.loaders",
            "agentic_rag.ingestion.chunking",
            "agentic_rag.ingestion.index",
            "agentic_rag.ingestion.download",
            "agentic_rag.rag",
            "agentic_rag.rag.state",
            "agentic_rag.rag.nodes",
            "agentic_rag.rag.graph",
        ),
        forbidden=heavy_modules(),
        # The fake embedding provider works without any model library.
        exercise=(
            "from agentic_rag.config import Settings\n"
            "from agentic_rag.embeddings import get_embeddings\n"
            "embeddings = get_embeddings(Settings(_env_file=None, embedding_provider='fake'))\n"
            "result = len(embeddings.embed_documents(['one text', 'another text']))\n"
        ),
        expected_result=2,
    ),
    ImportGroup(
        name="evaluation",
        # The light modules first, so that a graph-runtime import is pinned on the first one
        # that makes it.
        modules=(
            "agentic_rag.tracing",
            "agentic_rag.agent.types",
            "agentic_rag.reports",
            "agentic_rag.evaluation",
            "agentic_rag.evaluation.dataset",
            "agentic_rag.evaluation.metrics",
            "agentic_rag.evaluation.runner",
            "agentic_rag.loadtest",
            "agentic_rag.loadtest.runner",
        ),
        forbidden=heavy_modules(add=GRAPH_RUNTIME_MODULES),
    ),
    ImportGroup(
        name="ui",
        modules=("agentic_rag.ui", "agentic_rag.ui.components"),
        # The UI is the one place that may load Streamlit.
        forbidden=heavy_modules(allow={"streamlit"}),
        # The page as a browser first opens it, in fake mode.
        exercise=(
            "from pathlib import Path\n"
            "import agentic_rag.ui\n"
            "from streamlit.testing.v1 import AppTest\n"
            "app = Path(agentic_rag.ui.__file__).with_name('app.py')\n"
            "at = AppTest.from_file(str(app), default_timeout=60).run()\n"
            "result = [str(exception.value) for exception in at.exception]\n"
        ),
        expected_result=[],
        env={"LLM_PROVIDER": "fake", "EMBEDDING_PROVIDER": "fake"},
    ),
)


def _environment(group: ImportGroup, directory: Path) -> dict[str, str]:
    """The environment of the group's interpreter: no settings variables but the group's."""
    field_names = set(Settings.model_fields)
    env = {name: value for name, value in os.environ.items() if name.lower() not in field_names}
    env.update(
        {
            "DATA_DIR": str(directory / "data" / "raw"),
            "CHROMA_DIR": str(directory / "data" / "chroma_db"),
            "PYTHONIOENCODING": "utf-8",
            **group.env,
        }
    )
    return env


@pytest.fixture(scope="module")
def import_reports(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Run every group's probe in its own interpreter, all at the same time.

    Each interpreter starts in an empty working directory (no ``.env``) with none of the
    developer's settings variables.

    Yields:
        The report of each group by name, or a ``str`` with the interpreter's output if it
        printed no report.
    """
    processes: dict[str, subprocess.Popen[str]] = {}
    try:
        for group in GROUPS:
            directory = tmp_path_factory.mktemp(f"imports-{group.name}")
            (directory / "data" / "raw").mkdir(parents=True)
            spec = {
                "modules": group.modules,
                "forbidden": sorted(group.forbidden),
                "exercise": group.exercise,
            }
            processes[group.name] = subprocess.Popen(
                [sys.executable, "-c", _PROBE, json.dumps(spec)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=_environment(group, directory),
                cwd=directory,
            )
        reports: dict[str, Any] = {}
        for name, process in processes.items():
            stdout, stderr = process.communicate(timeout=120)
            lines = [line for line in stdout.splitlines() if line.startswith(REPORT_PREFIX)]
            if process.returncode == 0 and lines:
                reports[name] = json.loads(lines[-1].removeprefix(REPORT_PREFIX))
            else:
                reports[name] = f"exit code {process.returncode}\n{stdout}\n{stderr}"
        yield reports
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.communicate()


def _offences(report: dict[str, Any]) -> list[str]:
    """Describe every forbidden package in the report, naming the module that loaded it."""
    offences = []
    if report["at start-up"]:
        offences.append(f"the interpreter started with {', '.join(report['at start-up'])}")
    for module, loaded in report["imports"].items():
        if loaded:
            offences.append(f"importing {module} loaded {', '.join(loaded)}")
    if report["exercise"]:
        offences.append(f"the exercise loaded {', '.join(report['exercise'])}")
    return offences


@pytest.mark.parametrize("group", GROUPS, ids=[group.name for group in GROUPS])
def test_a_fresh_interpreter_loads_no_forbidden_module(
    group: ImportGroup, import_reports: dict[str, Any]
) -> None:
    report = import_reports[group.name]
    assert isinstance(report, dict), f"the {group.name} probe failed: {report}"

    offences = _offences(report)

    assert not offences, f"{group.name}: " + "; ".join(offences)
    assert list(report["imports"]) == list(group.modules)
    assert report["result"] == group.expected_result
