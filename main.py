"""Run the full training pipeline sequentially. Each step imports and calls its main() directly."""
from __future__ import annotations

import argparse
import sys
import time

STEPS = [
    ("eda",                    "training.scripts.eda"),
    ("preprocess",             "training.scripts.preprocess"),
    ("adversarial_validation", "training.scripts.adversarial_validation"),
    ("baselines",              "training.scripts.baselines"),
    ("tune",                   "training.scripts.tune"),
    ("evaluate",               "training.scripts.evaluate"),
    ("export_onnx",            "training.scripts.export_onnx"),
    ("load_features_to_redis", "training.scripts.load_features_to_redis"),
]

STEP_NAMES = [name for name, _ in STEPS]


def run_step(module_path: str) -> None:
    import importlib
    mod = importlib.import_module(module_path)
    mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fraud ML training pipeline")
    parser.add_argument(
        "--from",
        dest="from_step",
        choices=STEP_NAMES,
        default=None,
        help="Start from this step (inclusive). Defaults to the beginning.",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        choices=STEP_NAMES,
        default=None,
        help="Stop after this step (inclusive). Defaults to the end.",
    )
    parser.add_argument(
        "--only",
        dest="only_step",
        choices=STEP_NAMES,
        default=None,
        help="Run only this single step.",
    )
    args = parser.parse_args()

    steps = STEPS

    if args.only_step:
        steps = [(name, mod) for name, mod in STEPS if name == args.only_step]
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
            print(f"\nFailed at step {name}: {e}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"--- {name} done ({elapsed:.1f}s) ---\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
