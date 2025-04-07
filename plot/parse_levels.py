import os
import yaml
import re

def extract_level_number(path):
    # Look for pattern like "level_X" in the path
    match = re.search(r'level_(\d+)', path)
    if match:
        return int(match.group(1))
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
    for root in roots:
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
                    # if 'debug_prob' in config['science_runner_args']:
                    #     ipdb.set_trace()
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
                        }
                except Exception as e:
                    print(f"Error processing {config_path}: {e}")
    
    return folder_info

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
