"""Run from the project root after `uv sync`.

Usage:
  uv run python monitoring/main.py                          # run all steps
  uv run python monitoring/main.py --from train_challenger  # resume from a step
  uv run python monitoring/main.py --only register_models   # run one step

After this completes:
  docker compose --profile all up --build
"""
from __future__ import annotations

import argparse
import sys
import time

STEPS = [
    ("apply_migrations", "monitoring.scripts.apply_migrations"),
    ("train_challenger",  "monitoring.scripts.train_challenger"),
    ("register_models",   "monitoring.scripts.register_models"),
]

STEP_NAMES = [name for name, _ in STEPS]


def run_step(module_path: str) -> None:
    import importlib
    mod = importlib.import_module(module_path)
    mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 setup pipeline")
    parser.add_argument("--from", dest="from_step", choices=STEP_NAMES, default=None,
                        help="Start from this step (inclusive).")
    parser.add_argument("--to", dest="to_step", choices=STEP_NAMES, default=None,
                        help="Stop after this step (inclusive).")
    parser.add_argument("--only", dest="only_step", choices=STEP_NAMES, default=None,
                        help="Run only this single step.")
    args = parser.parse_args()

    steps = STEPS[:]

    if args.only_step:
        steps = [(n, m) for n, m in STEPS if n == args.only_step]
    else:
        if args.from_step:
            start = STEP_NAMES.index(args.from_step)
            steps = steps[start:]
        if args.to_step:
            end = STEP_NAMES.index(args.to_step) + 1
            steps = [(n, m) for n, m in steps if STEP_NAMES.index(n) < end]

    print(f"Running {len(steps)} step(s): {[n for n, _ in steps]}\n")

    for name, module_path in steps:
        print(f"--- {name} ---")
        t0 = time.time()
        try:
            run_step(module_path)
        except Exception as e:
            print(f"\nFailed at step '{name}': {e}", file=sys.stderr)
            sys.exit(1)
        print(f"--- {name} done ({time.time() - t0:.1f}s) ---\n")

    print("Monitoring setup complete.")
    print("")
    print("Next steps:")
    print("  1. Copy .env.example to .env and fill in REFERENCE_WINDOW_START,")
    print("     REFERENCE_WINDOW_END, CHAMPION_VERSION, CHALLENGER_VERSION.")
    print("     Values for the last two are in training/artifacts/model_versions.json.")
    print("  2. docker compose down (if stack is running)")
    print("  3. docker compose --profile monitoring up --build")


if __name__ == "__main__":
    main()
