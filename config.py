from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)
