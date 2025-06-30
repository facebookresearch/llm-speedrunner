# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch the LLM agent runs corresponding to the baselines of the paper

# In order to replicate the results, replace maui/maui_high with your actuall account/qos 
# and run the following commands.
# First, run for records 1-11 (we omit record 6 due to PyTorch upgrade):
conda activate record-1-11
python launchers/launch_llm_speedrun_agent_baselines.py \
--job_name baselines_records_1_11 \
--record_numbers 1 2 3 4 5 7 8 9 10 11 \
--knowledge_levels z 1 2 5 [12] [125] \
--ideator dummy 

# Then, run for records 12-18:
conda activate record-12-18
python launchers/launch_llm_speedrun_agent_baselines.py \
--job_name baselines_records_12_18 \
--record_numbers 12 13 14 15 16 17 18 \
--knowledge_levels z 1 2 5 [12] [125] \
--ideator dummy 

# Finally, for records 19-21:
conda activate record-19-21
python launchers/launch_llm_speedrun_agent_baselines.py \
--job_name baselines_records_19_20 \
--record_numbers 19 20 \
--knowledge_levels z 1 2 5 [12] [125] \
--ideator dummy 
"""

from typing import Optional
import os
import subprocess
import submitit
import argparse
import itertools
import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_cmd(
    record_number: int,
    model_name: str = "deepseek_r1",
    qos: str = "maui_high",
    n_iterations: int = 10,
    ideator: str = "dummy",
    science_runner: str = "bon",
    max_n_nodes: int = 20,
    n_hypotheses: int = 1,
    n_initial_hypotheses: int = 1,
    aide_debug_prob: float = 1.0,
    aide_max_bug_depth: int = 50,
    knowledge_level: str = "0",
    pass_coder_knowledge: bool = False,
    aider_edit_format: str = "diff",
    strict_diff_format: bool = False,
):
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
        f"coder_args.edit_format={aider_edit_format}",
        f"slurm_config_args.qos={qos}",
        f"science_runner_args.max_n_nodes={max_n_nodes}",
    ]

    if science_runner == 'bon':
        cmd.append(f"science_runner_args.n_hypotheses={n_hypotheses}")
        cmd.append(f"science_runner_args.n_initial_hypotheses={n_initial_hypotheses}")
    elif science_runner == 'aide':
        cmd.append(f"science_runner_args.n_initial_hypotheses={n_initial_hypotheses}")
        cmd.append(f"science_runner_args.n_hypotheses={n_hypotheses}")
        cmd.append(f"science_runner_args.debug_prob={aide_debug_prob}")
        cmd.append(f"science_runner_args.max_bug_depth={aide_max_bug_depth}")

    cmd.append(f'knowledge_src_paths=[{knowledge_path}]')
    
    if pass_coder_knowledge:
        cmd.append(f'science_runner_args.knowledge_pass_to_coder=True')

    return cmd

def get_slurm_id() -> str:
    slurm_id = []
    env_var_names = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"]
    for var_name in env_var_names:
        if var_name in os.environ:
            slurm_id.append(str(os.environ[var_name]))
            print(f"[DEBUG] Environment variable {var_name}: {str(os.environ[var_name])}")
    if slurm_id:
        return "-".join(slurm_id)
    return "-1"

def worker(cmd: list[str], workspace_path: Optional[str] = None):
    slurm_job_id = get_slurm_id()
    augmented_workspace_path = workspace_path + f"_{slurm_job_id}"
    cmd.append(f"workspace_args.root_path={augmented_workspace_path}")
    cwd = os.getcwd()
    for key, value in os.environ.items():
        print(f"[ENV] {key}: {value}")
    print("[INFO] Running in directory:", cwd)
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def set_agent_search_parameters(search, args):
    if search == "flat":
        args.science_runner = "bon"
        args.n_initial_hypotheses = 20
        args.n_iterations = 1
    elif search == "tree":
        args.science_runner = "bon"
        args.n_initial_hypotheses = 1
        args.n_hypotheses = 3
        args.n_iterations = 20
    elif search == "forest":
        args.science_runner = "bon"
        args.n_initial_hypotheses = 3
        args.n_hypotheses = 3
        args.n_iterations = 20
    elif search == "aide":
        args.science_runner = "aide"
        args.n_initial_hypotheses = 3
        args.n_hypotheses = 1
        args.n_iterations = 20
        args.aide_debug_prob = 0.5
        args.aide_max_bug_depth = 5
    elif search == "multi-aide":
        args.science_runner = "aide"
        args.n_initial_hypotheses = 3
        args.n_hypotheses = 3
        args.n_iterations = 20
        args.aide_debug_prob = 0.5
        args.aide_max_bug_depth = 5
    else:
        raise ValueError(f"Search {search} not valid")

def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for scientist jobs with knowledge source paths.")
    parser.add_argument("--job_name", type=str, default="knowledge_sweep", help="Job name")
    parser.add_argument("--qos", type=str, default="maui_high", help="Quality of service")
    parser.add_argument("--max_n_nodes", type=int, default=20, help="Maximum number of nodes to use")
    parser.add_argument("--record_numbers", type=int, nargs='+', required=True, help="List of record numbers to sweep over")
    parser.add_argument("--models", type=str, nargs='+', default=['deepseek_r1', 'o3_mini', 'gemini_2_5', 'claude_3_7_sonnet'], help="Frontier models powering the agent")
    parser.add_argument("--n_iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--ideator", type=str, default="dummy", help="Ideator to use")
    parser.add_argument("--science_runner", type=str, default="bon", help="Science runner to use")
    parser.add_argument("--n_hypotheses", type=int, default=3, help="Number of hypotheses")
    parser.add_argument("--n_initial_hypotheses", type=int, default=1, help="Number of initial hypotheses")
    parser.add_argument("--aide_debug_prob", type=float, default=1.0, help="Debug probability for AIDE")
    parser.add_argument("--aide_max_bug_depth", type=int, default=50, help="Max bug depth for AIDE")
    parser.add_argument("--knowledge_levels", nargs='+', default=['1', '2', '3', '[12]', '[123]'], type=str, help="Knowledge level to use in glob string format, e.g. 1 to only use level 1, {1,2} to use level 1 and 2, etc.")
    parser.add_argument("--array_parallelism", type=int, default=10, help="Number of jobs to run in parallel")
    parser.add_argument("--pass_coder_knowledge", type=str2bool, default=False, help="Whether or not to pass coder knowledge")
    parser.add_argument("--aider_edit_format", type=str, default="diff", help="Aider edit format")
    parser.add_argument("--strict_diff_format", type=str2bool, default=False, help="Whether or not to use strict diff format")
    parser.add_argument("--no_confirmation", type=str2bool, default=False, help="Whether or not to skip confirmation")
    parser.add_argument("--searches", type=str, nargs='+', default=['flat', 'tree', 'forest', 'aide', 'multi-aide'], help="Search variants to use")
    args = parser.parse_args()

    account = "maui"
    username = os.getlogin()
    root_workspace_path = f"/checkpoint/{account}/{username}/scientist/workspace/"
    executor = submitit.AutoExecutor(folder="submitit_logs/slurm_job_%j")
    executor.update_parameters(
            name=args.job_name,
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=8,
            timeout_min=6*24*60,  # 6 days
            slurm_array_parallelism=args.array_parallelism,
        )
    jobs = []
    iterator = list(itertools.product(
        args.record_numbers,
        args.knowledge_levels,
        args.searches,
        args.models,
    ))
    print(f"Generating {len(iterator)} commands")
    print(f"Root workspace path: {root_workspace_path}")
    
    for record_number, knowledge_level, search, model_name in iterator:
        set_agent_search_parameters(search, args)
        cmd = generate_cmd(
                record_number=record_number,
                model_name=model_name,
                n_iterations=args.n_iterations,
                ideator=args.ideator,
                science_runner=args.science_runner,
                n_hypotheses=args.n_hypotheses,
                n_initial_hypotheses=args.n_initial_hypotheses,
                aide_debug_prob=args.aide_debug_prob,
                aide_max_bug_depth=args.aide_max_bug_depth,
                knowledge_level=knowledge_level,
                pass_coder_knowledge=args.pass_coder_knowledge,
                aider_edit_format=args.aider_edit_format,
                strict_diff_format=args.strict_diff_format,
                max_n_nodes=args.max_n_nodes,
            )
        print(" ".join(cmd))

    if not args.no_confirmation:
        input("Press Enter to continue")
    
    with executor.batch():
        for record_number, knowledge_level, search, model_name in iterator:
            now = datetime.datetime.now()
            workspace_path_prefix = f"{root_workspace_path}record_{record_number}_{now:%Y%m%d_%H%M%S}"
            set_agent_search_parameters(search, args)
            job = executor.submit(
                worker,
                generate_cmd(
                    record_number=record_number,
                    model_name=model_name,
                    n_iterations=args.n_iterations,
                    ideator=args.ideator,
                    science_runner=args.science_runner,
                    n_hypotheses=args.n_hypotheses,
                    n_initial_hypotheses=args.n_initial_hypotheses,
                    aide_debug_prob=args.aide_debug_prob,
                    aide_max_bug_depth=args.aide_max_bug_depth,
                    knowledge_level=knowledge_level,
                    pass_coder_knowledge=args.pass_coder_knowledge,
                    aider_edit_format=args.aider_edit_format,
                    strict_diff_format=args.strict_diff_format,
                    max_n_nodes=args.max_n_nodes,
                ),
                workspace_path_prefix
            )
            jobs.append(job)

    for job in jobs:
        print("Job ID:", job.job_id, "State:", job.state)

if __name__ == "__main__":
    main()