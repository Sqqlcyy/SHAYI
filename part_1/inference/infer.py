
from __future__ import annotations

import sys
from pathlib import Path


PART1_ROOT = Path(__file__).resolve().parents[1]
if str(PART1_ROOT) not in sys.path:
    sys.path.insert(0, str(PART1_ROOT))

from evaluation.reconstruct import main  # noqa: E402


if __name__ == "__main__":
    main()
