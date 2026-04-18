"""
Run the full experiment pipeline in a single command with reproducibility metadata.

Stages:
1. Build processed datasets
2. Train baseline models
3. Run attacker simulation
4. Analyse simulation output
5. Generate result figures

Usage:
    python src/reproduce_pipeline.py
    python src/reproduce_pipeline.py --seed 42
    python src/reproduce_pipeline.py --skip-build-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_RESULTS_DIR = PROJECT_ROOT / "simulation_results"
METADATA_PATH = SIM_RESULTS_DIR / "reproducibility_run_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the full reproducible phishing-detection pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed exported as FYP_RANDOM_SEED and PYTHONHASHSEED. "
            "Current training/split logic in this repository uses seed=42 in code."
        ),
    )
    parser.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="Skip dataset build stage (assumes processed CSV files already exist).",
    )
    return parser.parse_args()


def build_stages(skip_build_dataset: bool) -> list[tuple[str, str]]:
    stages: list[tuple[str, str]] = []

    if not skip_build_dataset:
        stages.append(("build_dataset", "src/build_dataset.py"))

    stages.extend(
        [
            ("train_models", "src/mvp_baseline.py"),
            ("run_simulation", "src/attacker_sim.py"),
            ("analyse_results", "src/analyse_simulation_results.py"),
            ("visualise_results", "src/visualise_results.py"),
        ]
    )
    return stages


def run_stage(stage_name: str, script_path: str, env: dict[str, str]) -> float:
    print(f"\n{'=' * 70}")
    print(f"Stage: {stage_name}")
    print(f"Command: {sys.executable} {script_path}")
    print(f"{'=' * 70}")

    start = time.perf_counter()
    subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    elapsed_seconds = time.perf_counter() - start

    print(f"Completed stage '{stage_name}' in {elapsed_seconds:.2f}s")
    return elapsed_seconds


def write_metadata(
    seed: int,
    stages: list[tuple[str, str]],
    stage_durations: dict[str, float],
    started_at: str,
    finished_at: str,
) -> None:
    SIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "reproducibility": {
            "seed": seed,
            "pythonhashseed": str(seed),
            "env_variable": "FYP_RANDOM_SEED",
            "note": "Some scripts currently hardcode seed=42 internally; keep --seed=42 for identical reruns.",
        },
        "pipeline_stages": [
            {
                "name": stage_name,
                "script": script,
                "duration_seconds": round(stage_durations[stage_name], 3),
            }
            for stage_name, script in stages
        ],
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved reproducibility metadata to: {METADATA_PATH}")


def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["FYP_RANDOM_SEED"] = str(args.seed)

    stages = build_stages(skip_build_dataset=args.skip_build_dataset)

    started_at = datetime.now(timezone.utc).isoformat()
    stage_durations: dict[str, float] = {}

    print("Running full reproducible pipeline...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Seed: {args.seed}")

    for stage_name, script_path in stages:
        elapsed_seconds = run_stage(stage_name, script_path, env)
        stage_durations[stage_name] = elapsed_seconds

    finished_at = datetime.now(timezone.utc).isoformat()
    write_metadata(
        seed=args.seed,
        stages=stages,
        stage_durations=stage_durations,
        started_at=started_at,
        finished_at=finished_at,
    )

    total_seconds = sum(stage_durations.values())
    print(f"\nPipeline completed successfully in {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
