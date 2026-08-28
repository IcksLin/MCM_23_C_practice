"""Run the complete Q2 entry point with a small, explicitly uncertified grid."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable, str(root / "scripts" / "run_q2.py"),
        "--seed", "2024", "--raw-scenarios", "10",
        "--reduced-scenarios", "1", "--beta", "0.90",
        "--lambda-grid", "0:0:0.1", "--out-sample", "20",
        "--mip-gap", "0.8", "--time-limit", "10",
        "--allow-uncertified",
    ]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
