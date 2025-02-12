"""Automatically create the necessary config and workspace template for an MLE-Bench task.

Creates a task config at config/task/mlebench/<task id>.yaml, and creates a workspace template
at workspace_templates/mlebench/<task id>. 

In the generated artifacts, <task id> is the snake-case version of the MLE-Bench task ID.


IMPORTANT: In order for this script to run without any errors, you must first go to your 
MLE-Bench directory (from where you called its initial setup.py script), and comment out
the logging.info(...) line in data.py in the ensure_leaderboard_exists function. Their 
usage of relative_to to compute a relative path is not robust and will likely lead to an error.

Example usage:

python make_mlebench_task.py \
--task_id=random-acts-of-pizza \
--cache_dir_path=/checkpoint/maui/minqijiang/data

"""
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
    parser.add_argument('--n_preview_lines', type=int, default=20, help='Whether lower is better')
    args = parser.parse_args()

    task_id_safe = args.task_id.replace('-', '_')
    cache_dir_path = fs_utils.expand_path(args.cache_dir_path)
    config_dir_path = fs_utils.expand_path(args.config_dir_path)
    workspace_template_dir_path = fs_utils.expand_path(args.workspace_template_dir_path)
    task_template_path = os.path.join(workspace_template_dir_path, task_id_safe)

    # Run the prepare command to download task data to the specified cache dir
    prepare_command = f'XDG_CACHE_HOME={cache_dir_path} mlebench prepare -c {args.task_id} --overwrite-leaderboard'
    print(f'Running command: {prepare_command}')
    subprocess.run(prepare_command, shell=True, check=True)

    os.makedirs(config_dir_path, exist_ok=True)
    config_file_path = os.path.join(config_dir_path, f'{task_id_safe}.yaml')

    # Generate a preview of read-only public resource files for the agent
    public_data_path = f'{args.cache_dir_path}/mle-bench/data/{args.task_id}/prepared/public/'
    preview_files = [
        fs_utils.expand_path(os.path.join(public_data_path, f))
        for f in os.listdir(public_data_path)
        if os.path.isfile(os.path.join(public_data_path, f))
        and f.endswith(('.csv', '.json', '.jsonl', '.txt', '.md'))
        and not f.endswith('description.md')
    ]

    combined_preview = ''
    for fname in preview_files:
        with open(fname, 'r') as f:
            lines = f.read().split('\n')
            if lines:
                preview = '\n'.join(lines[:args.n_preview_lines])
                if len(lines) > args.n_preview_lines:
                    preview += '\n...'
                preview = f'# {os.path.basename(fname)}\n{preview}\n\n'
                combined_preview += preview

    combined_preview_file = os.path.join(task_template_path, 'preview_resources.txt')
    with open(combined_preview_file, 'w') as f:
        f.write(combined_preview)

    # task-specific config
    config_data = {
    	'defaults': [
    		'mlebench/default'
    	],
    	'template_dirname': f'mlebench/{task_id_safe}',
        'exp_config_args': {
            'lower_is_better': args.lower_is_better
        },
        'abs_read_only_fnames': [combined_preview_file],
        'slurm_config_args': {
            'env_vars': {
                f'PUBLIC_RESOURCE_PATH': f'{public_data_path}',
                f'GRADER_DATA_PATH': f'{args.cache_dir_path}/mle-bench/data'
            }
        }
    }

    with open(config_file_path, 'w') as config_file:
        config_file.write('# @package _global_\n\n')
        yaml.dump(config_data, config_file, default_flow_style=False)
    print(f'Created config file at: {config_file_path}')

    # ==== Set up workspace template and fill its contents ====
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
    # dst_sample_path = os.path.join(task_template_path, 'sampleSubmission.csv')

    shutil.copyfile(src_description, dst_description_path)
    print(f'Copied description.md to: {dst_description_path}')
    # shutil.copyfile(src_sample, dst_sample_path)
    # print(f'Copied sampleSubmission.csv to: {dst_sample_path}')


if __name__ == '__main__':
    main()
