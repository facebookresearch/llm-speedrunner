# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch cascaded scientist runs to measure training time reduction

This script implements a cascaded experiment where:
1. Start with record_1, run scientist for multiple settings
2. Pick best record_1' with max training time reduction
3. For records i = 1 through 20:
   - Start with record_i', run scientist for multiple settings
   - Pick best record_(i+1)' with max training time reduction
4. Compute average training time reduction across transitions
5. Identify where agent breaks down and gets no more speedups

conda activate record-1-11
python launchers/launch_cumulative_speedup.py \
--job_name cascaded_speedup 
"""
from typing import Optional, List, Dict, Tuple
import os
import subprocess
import submitit
import argparse
import itertools
import datetime
import json
import ntpath
import re
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


def gather_metrics(
    workspace_path: str, 
    metrics: list[str],
    workspace_template_path: Optional[str] = None
) -> pd.DataFrame:
    data = {'step': []}
    for m in metrics:
        data[m] = []

    step2path = {}

    if workspace_template_path is not None:
        initial_result_path = os.path.join(workspace_template_path, 'results.json')
        if os.path.exists(initial_result_path):
            step2path[0] = initial_result_path

    # Gather metrics from version subdirectories
    version_dir_pattern = re.compile(r"^v_(\d+)$")
    for entry in os.scandir(workspace_path):
        if entry.is_dir():
            match = version_dir_pattern.match(entry.name)
            if match:
                step = int(match.group(1))
                step2path[step] = os.path.join(entry.path, "results.json")

    for step, results_path in step2path.items():
        data['step'].append(step)
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                results_json = json.load(f)
            metrics_dict = results_json.get("metrics", {})
        else:
            metrics_dict = {}

        # Fill in metric values
        for m in metrics:
            value = metrics_dict.get(m, None)
            data[m].append(value)
    
    # check which solutions are valid and print them out
    for key, values in data.items():
        if key == 'is_valid':
            true_indices = [i for i, value in enumerate(values) if value]
            if true_indices:
                print(f"Indices of valid versions: {true_indices}")
                
    df = pd.DataFrame(data)
    df.sort_values(by="step", inplace=True, ignore_index=True)

    return df

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

cmd_template = {
    'debug': [
        'science_runner=bon',
        'science_runner_args.n_initial_hypotheses=1',
        'n_iterations=1',
    ],
    'flat': [
        'science_runner=bon',
        'science_runner_args.n_initial_hypotheses=20',
        'n_iterations=1',
    ],
    'tree': [
        'science_runner=bon',
        'science_runner_args.n_initial_hypotheses=1',
        'science_runner_args.n_hypotheses=3',
        'n_iterations=20',
    ],
    'forest': [
        'science_runner=bon',
        'science_runner_args.n_initial_hypotheses=3',
        'science_runner_args.n_hypotheses=3',
        'n_iterations=20',
    ],
    'aide': [
        'science_runner=aide',
        'science_runner_args.n_initial_hypotheses=3',
        'science_runner_args.n_hypotheses=1',
        'n_iterations=20',
        'science_runner_args.debug_prob=0.5',
        'science_runner_args.max_bug_depth=5',
    ],
    'multi-aide': [
        'science_runner=aide',
        'science_runner_args.n_initial_hypotheses=3',
        'science_runner_args.n_hypotheses=3',
        'n_iterations=20',
        'science_runner_args.debug_prob=0.5',
        'science_runner_args.max_bug_depth=5',
    ],
}

def generate_cmd(
    record_number: int,
    model_name: str = "deepseek_r1",
    qos: str = "maui_high",
    template: str = "flat",
    ideator: str = "dummy",
    science_runner: str = "bon",
    max_n_nodes: int = 20,
    knowledge_level: str = "123",  # Fixed to L123
    no_knowledge: bool = False,
    pass_coder_knowledge: bool = False,
    aider_edit_format: str = "diff",
    strict_diff_format: bool = False,
    workspace_template_path: Optional[str] = None,
):
    # wrap with ""
    knowledge_path = f'"data/nanogpt_speedrun_knowledge_in_levels/record_{record_number}/level_{knowledge_level}_*.txt"'
    
    cmd = [
        "python",
        f"launch_scientist.py",
        f"task=nanogpt_speedrun/speedrun_record_{record_number}",
        f"model={model_name}",
        f"ideator={ideator}",
        f"coder_args.edit_format={aider_edit_format}",
        f"slurm_config_args.qos={qos}",
        f"science_runner_args.max_n_nodes={max_n_nodes}",
    ]
    cmd.extend(cmd_template[template])

    if workspace_template_path is not None:
        cmd.append(f'workspace_args.template_dir={workspace_template_path}')
        cmd.append(f"slurm_config_args.job_name=cascaded_speedrun_{ntpath.basename(workspace_template_path)}")

    if not no_knowledge:
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

def get_training_time_reduction(workspace_path: str, current_record: int, keep_k: int = None) -> float:
    """Extract training time reduction from the workspace results"""
    metrics = gather_metrics(
        workspace_path, 
        ['train_time', 'val_loss'],
        workspace_template_path=os.path.join(
            'workspace_templates', 
            'nanogpt_speedrun', 
            f"record_{current_record}"
        )
    )
    if keep_k is not None:
        metrics = metrics[metrics['step'] <= keep_k]
    metrics.loc[metrics['val_loss'] >= 3.29, 'train_time'] = np.nan
    # check if the min train_time is smaller than 0.4 of the original train_time
    # if so, replace that with nan, and loop until the min is larger than 0.4 of the original train_time
    # if after replace and everything is nan, return 0.0
    while metrics['train_time'].min() < 0.4 * metrics['train_time'].iloc[0]:
        metrics.loc[metrics['train_time'].idxmin(), 'train_time'] = np.nan
    if metrics['train_time'].isna().all():
        return 0.0
    return metrics['train_time'].min()


def run_cascaded_experiment(
    model_names: List[str],
    templates: List[str],
    max_records: int = 21,
    array_parallelism: int = 10,
    dry_run: bool = False,
    break_at_failure: bool = False,
    failure_threshold: float = 0.5,
):
    """Run the cascaded experiment"""
    username = os.getlogin()
    root_workspace_path = f"/checkpoint/maui/{username}/scientist/workspace/test_cascaded_speedup/"
    if not dry_run:
        executor = submitit.AutoExecutor(folder="submitit_logs/slurm_job_%j")
        executor.update_parameters(
            name="cascaded_speedup",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=8,
            timeout_min=5*24*60,  # 5 days
            slurm_account="maui",
            slurm_qos="dev", # dev
            slurm_array_parallelism=array_parallelism,
        )

    # Store results for each transition
    results = []
    current_record = 1
    current_workspace_template_path = '/home/zhaobc/scientist/workspace_templates/nanogpt_speedrun/record_1'

    while current_record < max_records:
        print(f"\nStarting transition from record {current_record}")
        
        # Generate all combinations of settings
        settings = list(itertools.product(
            model_names,
            templates,
        ))
        
        jobs = []
        workspace_paths = []
        
        if not dry_run:
            # Submit jobs for all settings
            with executor.batch():
                for model_name, template in settings:
                    now = datetime.datetime.now()
                    workspace_path_prefix = f"{root_workspace_path}record_{current_record}_{now:%Y%m%d_%H%M%S}"
                    workspace_path_prefix += f"_{model_name}_{template}"
                    workspace_paths.append(workspace_path_prefix)
                    
                    job = executor.submit(
                        worker,
                        generate_cmd(
                            record_number=current_record,
                            model_name=model_name,
                            template=template,
                            workspace_template_path=current_workspace_template_path,
                        ),
                        workspace_path_prefix
                    )
                    jobs.append(job)

            # Wait for all jobs to complete
            for job in jobs:
                job.result()

            # Find best performing setting
            best_reduction = 0.0
            best_setting = None
            best_workspace = None

            for workspace_path, (model_name, template) in zip(workspace_paths, settings):
                reduction = get_training_time_reduction(workspace_path)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_setting = (model_name, template)
                    best_workspace = workspace_path

            if best_reduction <= 0.0:
                print(f"Warning: No speedup achieved at record {current_record}")
                break

            results.append({
                'from_record': current_record,
                'to_record': current_record + 1,
                'reduction': best_reduction,
                'settings': best_setting,
                'workspace': best_workspace
            })
            if best_reduction < failure_threshold and break_at_failure:
                print(f"[INFO] Steps to break down: Record {current_record} -> {current_record + 1}: {best_reduction:.2f}% reduction")
                break

            current_record += 1
            current_workspace_template_path = best_workspace
        else:
            cmds = []
            for model_name, template in settings:
                now = datetime.datetime.now()
                workspace_path_prefix = f"{root_workspace_path}record_{current_record}_{now:%Y%m%d_%H%M%S}"
                workspace_path_prefix += f"_{model_name}_{template}"
                workspace_paths.append(workspace_path_prefix)
                cmds.append(generate_cmd(
                    record_number=current_record,
                    model_name=model_name,
                    template=template,
                    workspace_template_path=current_workspace_template_path,
                ))
            print(f"Dry run, {len(cmds)} commands:")
            for cmd in cmds:
                print(" ".join(cmd))
            print("-" * 100)
            current_workspace_template_path = f'best_template_record_{current_record}'
            print(f"Dry run, skipping job submission, current_workspace_template_path: {current_workspace_template_path}")
            current_record += 1

    # Compute statistics
    if results:
        avg_reduction = sum(r['reduction'] for r in results) / len(results)
        print(f"\nAverage training time reduction: {avg_reduction:.2f}%")
        print("\nDetailed results:")
        for r in results:
            print(f"Record {r['from_record']} -> {r['to_record']}: {r['reduction']:.2f}% reduction")
            print(f"  Settings: {r['settings']}")
            print(f"  Workspace: {r['workspace']}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Submitit launcher for cascaded scientist jobs.")
    parser.add_argument("--job_name", type=str, default="cascaded_speedup", help="Job name")
    parser.add_argument("--model_names", type=str, nargs='+', default=["o3_mini"], help="List of model names to try")
    parser.add_argument("--templates", type=str, nargs='+', default=["flat", "tree", "forest", "aide", "multi-aide"], help="List of templates to try")
    parser.add_argument("--max_records", type=int, default=21, help="Maximum number of records to process")
    parser.add_argument("--array_parallelism", type=int, default=10, help="Number of jobs to run in parallel")
    parser.add_argument("--dry_run", type=str2bool, default=False, help="Whether or not to skip confirmation")
    parser.add_argument("--break_at_failure", type=str2bool, default=True, help="Whether or not to break at failure")
    parser.add_argument("--failure_threshold", type=float, default=0.5, help="Failure threshold")
    args = parser.parse_args()

    results = run_cascaded_experiment(
        model_names=args.model_names,
        templates=args.templates,
        max_records=args.max_records,
        array_parallelism=args.array_parallelism,
        dry_run=args.dry_run,
        break_at_failure=args.break_at_failure,
        failure_threshold=args.failure_threshold,
    )
    if not args.dry_run:
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"cascaded_speedup_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 