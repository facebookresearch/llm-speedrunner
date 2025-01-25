from typing import Optional, Type

import datetime
import dataclasses
import logging
import os
import shutil
import subprocess
import time

from scientist.core.experiment_runner import ExperimentRunner
import scientist.utils.shell_utils as shell_utils
import scientist.utils.fs_utils as fs_utils
import scientist.prompts as prompts


class NanoGPTClimber(ExperimentRunner):
	def run_exp(self):
		# See current solution
		code = self.workspace.view('train_gpt.py')

		# Request next hypothesis
		hypothesis = self.run_scientist(
			prompt.NANOGPT_TASK_GENERATE_HYPOTHESIS.format(code=code)
			validator=lambda x: validate_json(x, dict('hypothesis': str)),
		)['hypothesis']

		# Implement hypothesis
		updated_code = self.run_scientist(
			prompt.NANOGPT_TASK_IMPLEMENT_HYPOTHESIS.format(code=code, hypothesis=hypothesis)
		)

		# Save code to workspace's current version dir
		self.workspace.save_to_file(updated_code, 'train_gpt.py')

		# @todo: Launch experiment on slurm cluster + track jobid
		# @todo: bwrap the command
		job_id = slurm_utils.launch_job(
			command="train_gpt.py", 
			n_nodes=1, 
			gpus_per_node=8, 
			cpus_per_task=96, 
			tasks_per_node=1, 
			timeout_min=self.job_ttl,
			job_name='maui_climber',
			account='maui',
			qos='maui_high',
			working_dir=self.workspace.get_path()
		)

		# When time-limit exceeded, check for experiment results
		# @todo: Use slurm_utils.JobObserver to watch for job status and when done, runs callback
		time.sleep(self.job_ttl)
		job_out = slurm_utils.get_job_status(job_id)
		
		# Parse experiment results @todo: read results from logs
		fitness = 0
		if job_out:
			try:
				fitness = 1
			except:
				pass

		# Record experiment results in history
		# - Compute diff from previous code
		# - Store diffs in code
		# - Save result file in version_dir

		exp_metrics = dict(
			val_loss=val_loss,
			train_loss=train_loss,
			walltime=walltime
		)
		pass


def main():
	# Create scientist agent
	preamble = prompts.NANOGPT_CLIMBER_TASK_PREAMBLE
	scientist = Agent(model='qwen-r1-32b', system_prompt=prompts.SCIENTIST_SYSTEM_PROMPT)

	root_dir = 'workspaces/nanogpt' + f'_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}'
	workspace = Workspace(root_dir=root_dir)

	exp_config = ExperimentConfig(
		preamble=preamble,
		workspace=workspace, 
		scientist=scientist,
		job_ttl=10*60  # seconds
	)

	climber = NanoGPTClimber(workspace, scientist)

	climber.run(n_iterations=10)


if __name__ == '__main__':
	main()