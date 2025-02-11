'''Creates a <task id>.yaml file at config/task/mlebench for the specified MLE-Bench.

Usage:

python make_mlebench_task.py \
--task_id=random-acts-of-pizza \
--cache_dir_path=/checkpoint/maui/minqijiang/data/ \
--config_dir_path=config/task/mlebench \
--workspace_template_dir_path=workspace_templates/mlebench

'''
import argparse
import os
import json
import subprocess
import shutil
import yaml

from utils import fs_utils


def get_mlebench_file_path(task_id: str, cache_dir_path: str, filename: str):
	return os.path.join(
    	cache_dir_path, 'mle-bench', 'data', task_id, 'prepared', 'public', filename
    )


def main():
    parser = argparse.ArgumentParser(description='Prepare mlebench task workspace and config.')
    parser.add_argument('--task_id', type=str, required=True, help='Task id (in kebab-case)')
    parser.add_argument('--cache_dir_path', type=str, required=True, help='Cache directory path')
    parser.add_argument('--config_dir_path', type=str, default='config/task/mlebench', help='Config directory path')
    parser.add_argument('--workspace_template_dir_path', type=str, default='workspace_templates/mlebench', help='Workspace template directory path')
    parser.add_argument('--lower_is_better', action='store_true', default=False, help='Whether lower is better')
    args = parser.parse_args()

    task_id_safe = args.task_id.replace('-', '_')
    cache_dir_path = fs_utils.expand_path(args.cache_dir_path)
    config_dir_path = fs_utils.expand_path(args.config_dir_path)
    workspace_template_dir_path = fs_utils.expand_path(args.workspace_template_dir_path)

    # Run the prepare command to download task data to the specified cache dir
    prepare_command = f'XDG_CACHE_HOME={cache_dir_path} mlebench prepare -c {args.task_id} --overwrite-leaderboard'
    print(f'Running command: {prepare_command}')
    subprocess.run(prepare_command, shell=True, check=True)

    os.makedirs(config_dir_path, exist_ok=True)
    config_file_path = os.path.join(config_dir_path, f'{task_id_safe}.yaml')

    # task-specific config
    config_data = {
    	'defaults': [
    		'mlebench/default'
    	],
    	'template_dirname': f'mlebench/{task_id_safe}',
        'exp_config_args': {
            'lower_is_better': args.lower_is_better
        },
        'slurm_config_args': {
            'env_vars': {
                f'TRAIN_DATA_PATH': f'{args.cache_dir_path}/mle-bench/data/{args.task_id}/prepared/public/train.json',
                f'TEST_DATA_PATH': f'{args.cache_dir_path}/mle-bench/data/{args.task_id}/prepared/public/test.json',
                f'GRADER_DATA_PATH': f'{args.cache_dir_path}/mle-bench/data'
            }
        }
    }

    with open(config_file_path, 'w') as config_file:
        config_file.write('# @package _global_\n\n')
        yaml.dump(config_data, config_file, default_flow_style=False)
    print(f'Created config file at: {config_file_path}')

    # ==== Set up workspace template and fill its contents ====
    task_template_path = os.path.join(workspace_template_dir_path, task_id_safe)
    os.makedirs(task_template_path, exist_ok=True)

    # mlebench's grader expects this submission.jsonl to specify the submission file
    submission_jsonl = os.path.join(task_template_path, 'submission.jsonl')
    with open(submission_jsonl, 'w') as f:
    	json.dump(
    		dict(competition_id=args.task_id, submission_path='submission.csv'), f
    	)

    # Copy base files into task-specific workspace template
    fs_utils.cp_dir(os.path.join(workspace_template_dir_path, 'base'), task_template_path)

    src_description = get_mlebench_file_path(args.task_id, cache_dir_path, 'description.md')
    src_sample = get_mlebench_file_path(args.task_id, cache_dir_path, 'sampleSubmission.csv')
    
    dst_description_path = os.path.join(task_template_path, 'description.md')
    dst_sample_path = os.path.join(task_template_path, 'sampleSubmission.csv')

    shutil.copyfile(src_description, dst_description_path)
    print(f'Copied description.md to: {dst_description_path}')
    shutil.copyfile(src_sample, dst_sample_path)
    print(f'Copied sampleSubmission.csv to: {dst_sample_path}')


if __name__ == '__main__':
    main()
