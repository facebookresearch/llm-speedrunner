from datetime import datetime
from typing import Optional, Type
import dataclasses
import logging
import json
import os
import shutil
import subprocess
import time

from core.types import ExperimentConfig
from core.experiment_runner import ExperimentRunner
from core.workspace import Workspace
from core.agent import Agent
from core import validators
from utils import fs_utils
import prompts


def parse_nanogpt_logs(path: str):
	pass


class NanoGPTClimber(ExperimentRunner):
	def _run_exp(self, version: str):
		# See current solution
		code = self.workspace.view('train_gpt.py', version=version)

		# @todo: Read performance and summary from results.json

		# Request next hypothesis
		hypothesis_res = self.run_scientist(
			prompts.NANOGPT_TASK_GENERATE_HYPOTHESIS.format(code=code),
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
		)
		hypothesis = json.loads(hypothesis_res)['hypothesis']


		# Implement hypothesis
		updated_code = self.run_scientist(
			prompts.NANOGPT_TASK_IMPLEMENT_HYPOTHESIS.format(code=code, hypothesis=hypothesis)
		)

		# Save code to workspace's current version dir
		self.workspace.save_to_file(updated_code, 'train_gpt.py', version=version)

		# # @todo: Launch experiment on slurm cluster + track jobid
		# # @todo: bwrap the command
		# job_id = slurm_utils.launch_job(
		# 	command="train_gpt.py", 
		# 	n_nodes=1, 
		# 	gpus_per_node=8, 
		# 	cpus_per_task=96, 
		# 	tasks_per_node=1, 
		# 	timeout_min=self.job_ttl,
		# 	job_name='maui_climber',
		# 	account='maui',
		# 	qos='maui_high',
		# 	working_dir=self.workspace.resolve_path(version=version)
		# )

		# # When time-limit exceeded, check for experiment results
		# # @todo: Use slurm_utils.JobObserver to watch for job status and when done, runs callback
		# time.sleep(self.job_ttl)
		# job_out = slurm_utils.get_job_status(job_id)
		
		# # Parse experiment results @todo: read results from logs + save in results.json
		# fitness = 0
		# if job_out:
		# 	try:
		# 		fitness = 1
		# 	except:
		# 		pass

		# exp_metrics = dict(
		# 	val_loss=val_loss,
		# 	train_loss=train_loss,
		# 	walltime=walltime
		# )

	def run(self, n_iterations=1):
		for i in range(n_iterations):
			if i > 0:
				prev_version = str(i)
				version = self.workspace.create_version(from_version=prev_version)
			else:
				version = '1'
			self._run_exp(version=version)


def main():
	# Create scientist agent
	preamble = prompts.NANOGPT_TASK_PREAMBLE
	scientist = Agent(model='qwen-r1-32b', system_prompt=prompts.SCIENTIST_SYSTEM_PROMPT)

	root_path = 'workspaces/nanogpt' + f'_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}'
	template_dir = 'workspace_templates/nanogpt'
	workspace = Workspace(root_path=root_path, template_dir=template_dir)

	exp_config = ExperimentConfig(
		preamble=preamble,
		job_ttl=10*60  # seconds
	)

	climber = NanoGPTClimber(config=exp_config, workspace=workspace, scientist=scientist)

	climber.run(n_iterations=10)


if __name__ == '__main__':
	main()
