# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch single scientist run

python launch_single_scientist.py \
--job_name ngpt_100it
"""
from typing import Optional
import os
import subprocess
import submitit
import argparse

def run_scientist(
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)

    task_name = "collatz"
    model_name = "deepseek_r1"
    n_iterations = 5
    n_hypotheses = 5
    n_initial_hypotheses = 10
    debug_prob = 0.25
    max_bug_depth=5
    cmd = [
        "python",
        f"launch_scientist.py",
        f"task={task_name}",
        f"model={model_name}",
        f"n_iterations={n_iterations}",
        # change this to aide for aide runs
        f"science_runner=bon",
        f"exp_config_args.selection_metric=val_loss",
        f"exp_config_args.metrics_at_most=null",
    ]
    

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for scientist jobs.")
    parser.add_argument("--job_name", type=str, default="single_scientist", help="Job name")
    args = parser.parse_args()
    
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
            name=args.job_name or "only_one_scientist",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=32,
            timeout_min=3*24*60,  # Convert hours to minutes.
            slurm_account="maui",
            slurm_qos="maui_high",
        )
    job = executor.submit(
        run_scientist,
    )
    print("Submitted job with ID:", job.job_id)


if __name__ == "__main__":
    main()