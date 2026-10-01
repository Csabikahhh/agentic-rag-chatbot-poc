"""Entry point of ``python -m agentic_rag``; the commands are defined in ``agentic_rag.cli``."""

import sys

from agentic_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
