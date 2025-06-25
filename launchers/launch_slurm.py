# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch a batch of scientist runs.

Usage example:
```
python launch_slurm.py --job_name aide
```
"""
from typing import Optional
import os
import subprocess
import submitit
import argparse
import itertools


def run_scientist(
    task_name: "nanogpt_speedrun/record_1",
    model_name: str = "deepseek_r1",
    n_iterations=5,
    n_initial_hypotheses: int = 3,
    n_hypotheses: int = 1,
    debug_prob: float = 0.5,
    max_bug_depth: int = 3,
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)
    cmd = [
        "python",
        f"launch_scientist.py",
        f"task={task_name}",
        f"model={model_name}",
        f"n_iterations={n_iterations}",
        f"science_runner=aide",
        f"exp_config_args.selection_metric=val_loss",
        f"exp_config_args.metrics_at_most=null",
        f"science_runner_args.max_bug_depth={max_bug_depth}",
        f"science_runner_args.debug_prob={debug_prob}",
        f"science_runner_args.n_initial_hypotheses={n_initial_hypotheses}",
        f"science_runner_args.n_hypotheses={n_hypotheses}",
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for scientist jobs.")
    parser.add_argument(
        "--job_name", 
        type=str,
        default="scientist",
        help="Job name"
    )
    parser.add_argument("--timeout",
        type=int,
        default=1440,  # 24 hours
        help="Maximum job duration."
    )
    parser.add_argument(
        "--n_initial_hypotheses",
        type=int, nargs='+',
        default=[1, 3],
        help="Number of initial hypotheses tested (drafts)."
    )
    parser.add_argument("--n_hypotheses",
        type=int, nargs='+',
        default=[1, 3],
        help="List of number of hypotheses tested after the first search iteration (branching factor)."
    )
    parser.add_argument("--debug_prob",
        type=float, nargs='+',
        default=[0.25, 0.5],
        help="Probability of selecting a buggy node for debugging, rather than a non-buggy node for improvement."
    )
    parser.add_argument("--max_bug_depth",
        type=int, nargs='+',
        default=[1, 3],
        help="Maximum length allowed for a debug path (a sequence of all buggy nodes) in the search tree."
    )
    args = parser.parse_args()
    
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
            name=args.job_name,
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=32,
            timeout_min=args.timeout,  # Convert hours to minutes.
            array_parallelism=10,
        )
    jobs = []
    with executor.batch():
        iterator = itertools.product(
            args.n_hypotheses,
            args.n_initial_hypotheses,
            args.debug_prob,
            args.max_bug_depth,
        )

        for n_hypotheses, n_initial_hypotheses, debug_prob, max_bug_depth in iterator:
            job = executor.submit(
                run_scientist,
                n_hypotheses=n_hypotheses,
                n_initial_hypotheses=n_initial_hypotheses,
                debug_prob=debug_prob,
                max_bug_depth=max_bug_depth,
            )
            jobs.append(job)

    for job in jobs:
        print("Submitted Job ID:", job.job_id)


if __name__ == "__main__":
    main()