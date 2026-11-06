"""Print a Kaggle kernel log (a JSON list of {stream_name, time, data}) as plain text.

uv run python scripts/kaggle_log.py build/models_<stamp>/impactlens-train.log
"""

import json
import sys
from pathlib import Path


def log_text(raw: str) -> str:
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # already plain text
    return "".join(e.get("data", "") for e in entries)


if __name__ == "__main__":
    print(log_text(Path(sys.argv[1]).read_text()), end="")
