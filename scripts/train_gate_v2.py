from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_gate_v1 import main as train_gate_main  # noqa: E402


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def main() -> None:
    argv = sys.argv[1:]
    if "--include-score-features" not in argv:
        argv.append("--include-score-features")
    if not _has_option(argv, "--gate-name"):
        argv.extend(["--gate-name", "trained_gate_v2_cv"])
    if not _has_option(argv, "--output-dir"):
        argv.extend(["--output-dir", "data/scored/gate_v2"])
    if not _has_option(argv, "--weight-power"):
        argv.extend(["--weight-power", "4"])

    sys.argv = [sys.argv[0], *argv]
    train_gate_main()


if __name__ == "__main__":
    main()
