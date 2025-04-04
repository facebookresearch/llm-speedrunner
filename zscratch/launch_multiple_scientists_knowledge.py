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
    bon_n_hypotheses: int = 1,
    aide_n_initial_hypotheses: int = 1,
    aide_n_hypotheses: int = 1,
    aide_debug_prob: float = 1.0,
    aide_max_bug_depth: int = 50,
    knowledge_level: int = 0,
    multiple_knowledge_paths: bool = False,
    no_knowledge: bool = False,
):
    cwd = os.getcwd()
    print("[INFO] Running in directory:", cwd)

    if multiple_knowledge_paths:
        knowledge_paths = []
        for i in range(knowledge_level+1):
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
    ]

    if science_runner == 'bon':
        cmd.append(f"science_runner_args.n_hypotheses={bon_n_hypotheses}")
    elif science_runner == 'aide':
        cmd.append(f"science_runner_args.n_initial_hypotheses={aide_n_initial_hypotheses}")
        cmd.append(f"science_runner_args.n_hypotheses={aide_n_hypotheses}")
        cmd.append(f"science_runner_args.debug_prob={aide_debug_prob}")
        cmd.append(f"science_runner_args.max_bug_depth={aide_max_bug_depth}")

    if not no_knowledge:
        cmd.append(f'knowledge_src_paths=[{knowledge_path}]')

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
    parser.add_argument("--bon_n_hypotheses", type=int, default=3, help="Number of hypotheses for BON")
    parser.add_argument("--aide_n_initial_hypotheses", type=int, default=1, help="Number of initial hypotheses for AIDE")
    parser.add_argument("--aide_n_hypotheses", type=int, default=1, help="Number of hypotheses for AIDE")
    parser.add_argument("--aide_debug_prob", type=float, default=1.0, help="Debug probability for AIDE")
    parser.add_argument("--aide_max_bug_depth", type=int, default=50, help="Max bug depth for AIDE")
    parser.add_argument("--knowledge_level", type=int, nargs='+', default=[0, 1, 2, 3, 4], help="Knowledge level to use")
    parser.add_argument("--multiple_knowledge_paths", type=bool, default=False, help="Whether to use multiple knowledge paths")
    parser.add_argument("--array_parallelism", type=int, default=10, help="Number of jobs to run in parallel")
    parser.add_argument("--no_knowledge", type=bool, default=False, help="Whether or not no knowledge")
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
            array_parallelism=args.array_parallelism,
        )
    jobs = []
    if args.no_knowledge:
        args.knowledge_level = [-1]
    
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
                bon_n_hypotheses=args.bon_n_hypotheses,
                aide_n_initial_hypotheses=args.aide_n_initial_hypotheses,
                aide_n_hypotheses=args.aide_n_hypotheses,
                aide_debug_prob=args.aide_debug_prob,
                aide_max_bug_depth=args.aide_max_bug_depth,
                knowledge_level=knowledge_level,
                multiple_knowledge_paths=args.multiple_knowledge_paths,
                no_knowledge=args.no_knowledge,
            )
            jobs.append(job)

    for job in jobs:
        print("Job ID:", job.job_id, "State:", job.state)

if __name__ == "__main__":
    main()
