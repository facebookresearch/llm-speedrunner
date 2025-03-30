"""Launch multiple scientist runs with different knowledge source paths

python zscratch/launch_multiple_scientists_knowledge.py \
--job_name knowledge_sweep \
--record_numbers 1 2 3 4 5 6 7 8 9 \
--model_name deepseek_r1 \
--n_iterations 10
"""
from typing import Optional
import os
import subprocess
import submitit
import argparse
import itertools
import ipdb
def run_scientist_with_knowledge(
    record_number: int,
    model_name: str = "deepseek_r1",
    n_iterations: int = 10,
    ideator: str = "dummy",
    science_runner: str = "bon",
    knowledge_level: int = 0,
    multiple_knowledge_paths: bool = False,
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)

    if multiple_knowledge_paths:
        knowledge_paths = []
        for i in range(knowledge_level):
            knowledge_paths.append(f"data/nanogpt_speedrun_knowledge_in_levels/record_{record_number}/level_{i}_*.txt")
        # join by , and ""
        knowledge_path = ",".join(f'"{path}"' for path in knowledge_paths)
    else:
        # wrap with ""
        knowledge_path = f'"data/nanogpt_speedrun_knowledge_in_levels/record_{record_number}/level_{knowledge_level}_*.txt"'
    cmd = [
        "python",
        f"launch_scientist.py",
        f"task=nanogpt_speedrun/speedrun_record_{record_number}",
        f"model={model_name}",
        f"n_iterations={n_iterations}",
        f"ideator={ideator}",
        f"science_runner={science_runner}",
        f'knowledge_src_paths={knowledge_path}',
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for scientist jobs with knowledge source paths.")
    parser.add_argument("--job_name", type=str, default="knowledge_sweep", help="Job name")
    parser.add_argument("--record_numbers", type=int, nargs='+', required=True, help="List of record numbers to sweep over")
    parser.add_argument("--model_name", type=str, default="deepseek_r1", help="Model name")
    parser.add_argument("--n_iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--ideator", type=str, default="dummy", help="Ideator to use")
    parser.add_argument("--science_runner", type=str, default="bon", help="Science runner to use")
    parser.add_argument("--knowledge_level", type=int, nargs='+', default=[0, 1, 2, 3, 4], help="Knowledge level to use")
    parser.add_argument("--multiple_knowledge_paths", type=bool, default=False, help="Whether to use multiple knowledge paths")
    args = parser.parse_args()
    
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
            name=args.job_name,
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=32,
            timeout_min=3*24*60,  # 3 days
            slurm_account="maui",
            slurm_qos="maui_high",
        )
    jobs = []
    
    with executor.batch():
        iterator = itertools.product(
            args.record_numbers,
            args.knowledge_level,
        )
        for record_number, knowledge_level in iterator:
            job = executor.submit(
                run_scientist_with_knowledge,
                record_number=record_number,
                model_name=args.model_name,
                n_iterations=args.n_iterations,
                ideator=args.ideator,
                science_runner=args.science_runner,
                knowledge_level=knowledge_level,
                multiple_knowledge_paths=args.multiple_knowledge_paths,
            )
            jobs.append(job)

    for job in jobs:
        print("Job ID:", job.job_id, "State:", job.state)

if __name__ == "__main__":
    main() 