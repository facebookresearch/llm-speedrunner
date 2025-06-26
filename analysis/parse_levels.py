# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
import re
import numpy as np
from collections import defaultdict

def extract_level_number(path):
    # Look for pattern like "level_X" in the path
    match = re.search(r'level_(\d+)', path)
    if match:
        return int(match.group(1))
    # also look for pattern like "level_[12]"
    match = re.search(r'level_\[(\d+)\]', path)
    if match:
        return int(match.group(1))
    # also look for pattern like "level_[z]"
    match = re.search(r'level_\[z\]', path)
    if match:
        return 'z'
    return None

def extract_record_number(directory_name):
    # Look for pattern like "record_X_" in the directory name
    match = re.search(r'record_(\d+)_', directory_name)
    if match:
        return int(match.group(1))
    return None

def find_levels_in_configs(base_dir):
    folder_info = {}
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        if 'config.yaml' in files:
            # Get the immediate parent directory name
            current_dir = os.path.basename(root)
            record_num = extract_record_number(current_dir)
            
            if record_num is None:
                continue
                
            config_path = os.path.join(root, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Check if knowledge_src_paths exists
                    if 'knowledge_src_paths' in config:
                        paths = config['knowledge_src_paths']
                        if isinstance(paths, str):
                            paths = [paths]
                            
                        all_levels = []
                        for path in paths:
                            level = extract_level_number(path)
                            if level is not None:
                                all_levels.append(level)
                        if len(all_levels) > 0:
                            folder_info[current_dir] = {
                                'record': record_num,
                                'levels': all_levels
                            }
            except Exception as e:
                print(f"Error processing {config_path}: {e}")
    
    return folder_info

from tqdm import tqdm

def find_levels_in_configs_glob(glob_strs):
    import glob
    folder_info = {}
    if not isinstance(glob_strs, list):
        glob_strs = [glob_strs]
    roots = []
    for i, glob_str in enumerate(glob_strs):
        roots += glob.glob(glob_str)
    print(f"Found {len(roots)} directories")
    
    # Walk through all directories
    for root in tqdm(roots):
        for _, dirs, files in os.walk(root):
            if 'config.yaml' in files:
                # Get the immediate parent directory name
                current_dir = os.path.basename(root)
                record_num = extract_record_number(current_dir)
                
                if record_num is None:
                    continue
                    
                config_path = os.path.join(root, 'config.yaml')
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        
                    all_levels = []
                    ideator = ''
                    if 'knowledge_pass_to_coder' in config['science_runner_args']:
                        if config['science_runner_args']['knowledge_pass_to_coder']:
                            knowledge_coder = True
                        else:
                            knowledge_coder = False
                    elif 'knowledge' in config['coder_args']['_target_']:
                        knowledge_coder = True
                    else:
                        knowledge_coder = False
                        
                    if 'debug_prob' in config['science_runner_args']:
                        runner = 'aide'
                    else:
                        runner = 'bon'

                    n_initial_hypotheses = config['science_runner_args'].get('n_initial_hypotheses', None)
                    n_hypotheses = config['science_runner_args'].get('n_hypotheses', None)
                    n_iterations = config['n_iterations']

                    # Check if knowledge_src_paths exists
                    if 'knowledge_src_paths' in config:
                        paths = config['knowledge_src_paths']
                        if isinstance(paths, str):
                            paths = [paths]
                            
                        for path in paths:
                            level = extract_level_number(path)
                            if level is not None:
                                all_levels.append(level)

                    if 'ideator_args' in config:
                        if 'dummy' in config['ideator_args']['_target_']:
                            ideator = 'dummy'
                        else:
                            ideator = 'base'

                    if len(all_levels) > 0:
                        folder_info[current_dir] = {
                            'record': record_num,
                            'levels': all_levels,
                            'ideator': ideator,
                            'knowledge_coder': knowledge_coder,
                            'runner': runner,
                            'model': config['model_name'],
                            'n_initial_hypotheses': n_initial_hypotheses,
                            'n_hypotheses': n_hypotheses,
                            'debug_prob': None if runner != 'aide' else config['science_runner_args']['debug_prob'],
                            'n_iterations': n_iterations,
                        }
                except Exception as e:
                    print(f"Error processing {config_path}: {e}")
    
    return folder_info

def filter_folder_info(folder_info, conditions):
    # folder_info is a dictionary of the form {folder_name: {info}}
    # conditions is a list of tuples, each tuple contains a condition
    # the format of each condition is:
    # (key, value)
    # when all conditions are met, the folder is added to the filtered_folder_info
    filtered_folder_info = {}
    for folder_name, info in folder_info.items():
        all_conditions_met = True
        for condition in conditions:
            if condition[0] == 'levels':
                cond_info = info['levels'][0]
            else:
                cond_info = info[condition[0]]

            if cond_info != condition[1]:
                all_conditions_met = False
                break
        if all_conditions_met:
            filtered_folder_info[folder_name] = info
    return filtered_folder_info

workspace_base_path = lambda item: os.path.join('/checkpoint/maui/zhaobc/scientist/workspace', item)
old_workspace_base_path = lambda item: os.path.join('/checkpoint/maui/zhaobc/scientist/old_workspace/nanogpt_speedrun/', item)
list_of_base_paths = [
    '/checkpoint/maui/zhaobc/scientist/workspace', 
    '/checkpoint/maui/zhaobc/scientist/old_workspace/nanogpt_speedrun/',
    '/checkpoint/maui/zhaobc/scientist/workspace/0507_relaunch/',
    '/checkpoint/maui/zhaobc/scientist/workspace/0511_relaunch/',
    '/checkpoint/maui/zhaobc/scientist/workspace/r1_relaunch/',
]
from plot_utils import gather_metrics

def process_metrics(record, workspace_base_path=workspace_base_path, gather_metrics=gather_metrics, keep_k=None):
    for k, v in record.items():
        try:
            metrics = gather_metrics(
                workspace_base_path(k),
                metrics=['val_loss', 'train_time'],
                workspace_template_path=os.path.join(
                    'workspace_templates', 
                    'nanogpt_speedrun', 
                    f"record_{v['record']}"
                )
            )
        except Exception as e:
            exist_path = None
            for base_path in list_of_base_paths:
                if os.path.exists(os.path.join(base_path, k)):
                    exist_path = base_path
                    break
            if exist_path is None:
                raise ValueError(f"Path {k} does not exist in any of the base paths")
            metrics = gather_metrics(
                exist_path,
                metrics=['val_loss', 'train_time'],
                workspace_template_path=os.path.join(
                    'workspace_templates', 
                    'nanogpt_speedrun', 
                    f"record_{v['record']}"
                )
            )
        if keep_k is not None:
            # only keep the first keep_k metrics
            metrics = metrics[metrics['step'] <= keep_k]
        metrics.loc[metrics['val_loss'] >= 3.29, 'train_time'] = np.nan
        record[k]['metrics'] = metrics
    return record

def process_metrics_raw_path(raw_path, record_num, gather_metrics=gather_metrics, keep_k=None):
    metrics = gather_metrics(
        raw_path,
        metrics=['val_loss', 'train_time'],
        workspace_template_path=os.path.join(
            'workspace_templates', 
                'nanogpt_speedrun', 
                f"record_{record_num}"
        )
    )
    if keep_k is not None:
        # only keep the first keep_k metrics
        metrics = metrics[metrics['step'] <= keep_k]
    metrics.loc[metrics['val_loss'] >= 3.29, 'train_time'] = np.nan
    return metrics


human_train_time_dict = {
    1: 2968348,
    2: 2209926,
    3: 1386147,
    4: 1301740,
    5: 949528,
    6: 766259,
    7: 773072,
    8: 662205,
    9: 505531,
    10: 477150,
    11: 442985,
    12: 317839,
    13: 289805,
    14: 273107,
    15: 241463,
    16: 232971,
    17: 220374,
    18: 211840,
    19: 199442,
    20: 188680,
    21: 184262,
}

def convert_to_dict(record, keep_dim=False, keep_name=False):
    results = {}
    for k, v in record.items():
        # record_num = int(k.split('-')[-1])
        ## the +1 here is because the process is 0-indexed but record number is 1-indexed
        # results[record_num + 1] = v['metrics']['train_time'].min()
        # record_num = k.split('_2025')[0].split('_')[-1]
        pattern = r"^record_(\d+)_"
        match = re.match(pattern, k)
        record_num = int(match.group(1))
        if keep_dim:
            to_save = v['metrics']['train_time'].tolist()
        else:
            to_save = v['metrics']['train_time'].min()
        if keep_name:
            to_save = (k, to_save)
        results[record_num] = to_save
    return results

def convert_to_dict_multiple_runs(record, keep_dim=False, keep_name=False):
    results = defaultdict(list)
    for k, v in record.items():
        # record_num = int(k.split('-')[-1])
        ## the +1 here is because the process is 0-indexed but record number is 1-indexed
        # results[record_num + 1] = v['metrics']['train_time'].min()
        # record_num = k.split('_2025')[0].split('_')[-1]
        pattern = r"^record_(\d+)_"
        match = re.match(pattern, k)
        record_num = int(match.group(1))
        if keep_dim:
            to_save = v['metrics']['train_time'].tolist()
        else:
            to_save = v['metrics']['train_time'].min()
        if keep_name:
            to_save = (k, to_save)
        results[record_num].append(to_save)
    return results

def compute_gap_in_percentage(
        model_time, 
        human_time=human_train_time_dict, 
        keep_name=False,
    ):
    gaps = {}
    for k, v in human_time.items():
        if (k + 1) not in human_time:
            continue
        gaps[k] = v - human_time[k+1]
    
    recovered_times = {}
    for k, v in model_time.items():
        if keep_name:
            recovered_time = human_time[k] - v[1]
        else:
            recovered_time = human_time[k] - v
        if keep_name:
            recovered_times[k] = (v[0], recovered_time)
        else:
            recovered_times[k] = recovered_time

    recovered_gap_in_percentage = {}
    for k, v in recovered_times.items():
        if (k + 1) not in gaps:
            continue
        if keep_name:
            recovered_gap_in_percentage[k] = v[1] / gaps[k] if gaps[k] > 0 else 0
        else:
            recovered_gap_in_percentage[k] = v / gaps[k] if gaps[k] > 0 else 0

    return recovered_gap_in_percentage

def compute_gap_in_percentage_list_keep_name(
        model_time, 
        human_time=human_train_time_dict, 
        keep_name=False,
    ):
    gaps = {}
    for k, v in human_time.items():
        if (k + 1) not in human_time:
            continue
        gaps[k] = v - human_time[k+1]
    
    recovered_times = {}
    for k, vs in model_time.items():
        recovered_times[k] = []
        for v in vs:
            if keep_name:
                recovered_time = human_time[k] - v[1]
            else:
                recovered_time = human_time[k] - v
            if keep_name:
                recovered_times[k].append((v[0], recovered_time))
            else:
                recovered_times[k].append(recovered_time)

    recovered_gap_in_percentage = {}
    for k, vs in recovered_times.items():
        if (k + 1) not in gaps:
            continue
        recovered_gap_in_percentage[k] = []
        for v in vs:
            if keep_name:
                recovered_gap_in_percentage[k].append((v[0], v[1] / gaps[k] if gaps[k] > 0 else 0))
            else:
                recovered_gap_in_percentage[k].append(v / gaps[k] if gaps[k] > 0 else 0)
        if keep_name:
            value = np.array([v[1] for v in recovered_gap_in_percentage[k]])
            value[np.isnan(value)] = 0
            name = np.array([v[0] for v in recovered_gap_in_percentage[k]])
            recovered_gap_in_percentage[k] = (name, value)
        else:
            recovered_gap_in_percentage[k] = np.array(recovered_gap_in_percentage[k])
            # replace nan with 0
            recovered_gap_in_percentage[k][np.isnan(recovered_gap_in_percentage[k])] = 0

    return recovered_gap_in_percentage

def compute_gap_in_percentage_list(
        model_time, 
        human_time=human_train_time_dict, 
        keep_name=False,
    ):
    gaps = {}
    for k, v in human_time.items():
        if (k + 1) not in human_time:
            continue
        gaps[k] = v - human_time[k+1]
    
    recovered_times = {}
    for k, vs in model_time.items():
        recovered_times[k] = []
        for v in vs:
            if keep_name:
                recovered_time = human_time[k] - v[1]
            else:
                recovered_time = human_time[k] - v
            if keep_name:
                recovered_times[k].append((v[0], recovered_time))
            else:
                recovered_times[k].append(recovered_time)

    recovered_gap_in_percentage = {}
    for k, vs in recovered_times.items():
        if (k + 1) not in gaps:
            continue
        recovered_gap_in_percentage[k] = []
        for v in vs:
            if keep_name:
                recovered_gap_in_percentage[k].append(v[1] / gaps[k] if gaps[k] > 0 else 0)
            else:
                recovered_gap_in_percentage[k].append(v / gaps[k] if gaps[k] > 0 else 0)

        recovered_gap_in_percentage[k] = np.array(recovered_gap_in_percentage[k])
        # replace nan with 0
        recovered_gap_in_percentage[k][np.isnan(recovered_gap_in_percentage[k])] = 0

    return recovered_gap_in_percentage



if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    base_dir = "workspaces/nanogpt_speedrun"
    folder_info = find_levels_in_configs_glob(
        [
            '/checkpoint/maui/zhaobc/scientist/workspace/nanogpt_speedrun/record_*_20250404_*',
            '/checkpoint/maui/zhaobc/scientist/workspace/nanogpt_speedrun/record_*_20250405_*'
        ]
    )
    print("Folder information:")
    # for folder_name, info in sorted(folder_info.items()):
    #     print(f"{folder_name}:")
    #     print(f"  Record: {info['record']}")
    #     print(f"  Levels: {info['levels']}")
    #     print(f"  Ideator: {info['ideator']}")
    #     print(f"  Knowledge coder: {info['knowledge_coder']}")
    #     print(f"  Runner: {info['runner']}")
    #     print(f"  Model: {info['model']}")

    # print r1, bon, dummy ideator, coder no knowledge
    for folder_name, info in folder_info.items():
        if info['model'] == 'deepseek-r1':
            if info['runner'] == 'bon':
                if info['ideator'] == 'dummy':
                    print(f"{folder_name}: r1, bon, dummy, no knowledge")
    
    print("--" * 10)
    # print o3, bon, dummy ideator, coder no knowledge
    for folder_name, info in folder_info.items():
        if info['model'] == 'o3-mini':
            if info['runner'] == 'bon':
                if info['ideator'] == 'dummy':
                    print(f"{folder_name}: o3, bon, dummy, no knowledge")
    
    print("--" * 10)
    # print r1, aide, dummy ideator, coder no knowledge
    for folder_name, info in folder_info.items():
        if info['model'] == 'deepseek-r1':
            if info['runner'] == 'aide':
                if info['ideator'] == 'dummy':
                    print(f"{folder_name}: r1, aide, dummy, no knowledge")

    print("--" * 10)
    # print r1, aide, base ideator, coder knowledge
    for folder_name, info in folder_info.items():
        if info['model'] == 'deepseek-r1':
            if info['runner'] == 'aide':
                if info['ideator'] == 'base':
                    if info['knowledge_coder']:
                        print(f"{folder_name}: r1, aide, base, knowledge")
