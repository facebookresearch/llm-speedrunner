# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch multiple scientist runs

python zscratch/launch_multiple_scientists.py \
--job_name aide_bon_runs 
"""
from typing import Optional
import os
import subprocess
import submitit
import argparse
import itertools

def run_scientist_bon(
    n_hypotheses: int = 1
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)

    task_name = "nanogpt_10112024"
    model_name = "deepseek_r1"
    n_iterations = 100
    cmd = [
        "python",
        f"launch_scientist.py",
        f"task={task_name}",
        f"model={model_name}",
        f"n_iterations={n_iterations}",
        f"science_runner=bon",
        f"exp_config_args.selection_metric=val_loss",
        f"exp_config_args.metrics_at_most=null",
        f"science_runner_args.n_hypotheses={n_hypotheses}",
    ]
    

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_scientist_aide(
    n_hypotheses: int = 1,
    n_initial_hypotheses: int = 7,
    debug_prob: float = 0.5,
    max_bug_depth: int = 3
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)

    task_name = "nanogpt_10112024"
    model_name = "deepseek_r1"
    n_iterations = 3
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
    parser.add_argument("--job_name", type=str, default="many_scientists", help="Job name")
    parser.add_argument("--bon_n_hypotheses", type=int, nargs='+', default=[1, 3, 5], help="List of number of hypothese for bon science runner to sweep over.")
    parser.add_argument("--aide_n_hypotheses", type=int, nargs='+', default=[1, 3], help="List of number of hypothese for aide science runner to sweep over.")
    parser.add_argument("--aide_n_initial_hypotheses", type=int, nargs='+', default=[5, 7], help="List of number of hypothese for bon science runner to sweep over.")
    parser.add_argument("--aide_debug_prob", type=float, nargs='+', default=[0.25, 0.5], help="List of number of hypothese for bon science runner to sweep over.")
    parser.add_argument("--aide_max_bug_depth", type=int, nargs='+', default=[3, 5], help="List of number of hypothese for bon science runner to sweep over.")
    args = parser.parse_args()
    
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
            name=args.job_name or "very_many_scientists",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=32,
            timeout_min=3*24*60,  # Convert hours to minutes.
            slurm_account="maui",
            slurm_qos="maui_high",
            array_parallelism=2,
        )
    jobs = []
    with executor.batch():
        for n_hypotheses in args.bon_n_hypotheses:
            job = executor.submit(
                run_scientist_bon,
                n_hypotheses=n_hypotheses,
            )
            jobs.append(job)
        
        iterator = itertools.product(
            args.aide_n_hypotheses,
            args.aide_n_initial_hypotheses,
            args.aide_debug_prob,
            args.aide_max_bug_depth,
        )

        for n_hypotheses, n_initial_hypotheses, debug_prob, max_bug_depth in iterator:
            job = executor.submit(
                run_scientist_aide,
                n_hypotheses=n_hypotheses,
                n_initial_hypotheses=n_initial_hypotheses,
                debug_prob=debug_prob,
                max_bug_depth=max_bug_depth,
            )
            jobs.append(job)

    for job in jobs:
        print("Job ID:", job.job_id, "State:", job.state)

if __name__ == "__main__":
    main()