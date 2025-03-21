"""Launch multiple scientist runs

python zscratch/launch_multiple_scientists.py \
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

    task_name = "nanogpt_10112024"
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
        # leaving these here for future test runs
        # f"science_runner_args.max_bug_depth={max_bug_depth}",
        # f"science_runner_args.debug_prob={debug_prob}",
        # f"science_runner_args.n_initial_hypotheses={n_initial_hypotheses}",
        # f"science_runner_args.n_hypotheses={n_hypotheses}",
    ]
    

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for scientist jobs.")
    parser.add_argument("--job_name", type=str, default="many_scientists", help="Job name")
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
        )
    job = executor.submit(
        run_scientist,
    )
    print("Submitted job with ID:", job.job_id)


if __name__ == "__main__":
    main()