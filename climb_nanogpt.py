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
from utils import slurm_utils
import prompts


class NanoGPTClimber(ExperimentRunner):
	async def _run_exp(self, version: str):
		# See current solution
		code = self.workspace.view('train_gpt.py', version=version)
		results = json.loads(self.workspace.view('results.json', version=version))

		# Request next hypothesis
		hypothesis_res = self.run_scientist(
			prompts.NANOGPT_TASK_GENERATE_HYPOTHESIS.format(code=code, results=results),
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
		)
		hypothesis = json.loads(hypothesis_res)['hypothesis']

		# Implement hypothesis
		updated_code = self.run_scientist(
			prompts.NANOGPT_TASK_IMPLEMENT_HYPOTHESIS.format(code=code, hypothesis=hypothesis)
		)

		# Save code to workspace's current version dir
		self.workspace.save_to_file(updated_code, 'train_gpt.py', version=version)

		# Launch and observe the job 
		job = slurm_utils.launch_job(
			command="train_gpt.py",
			bwrap=True, 
			n_nodes=1, 
			gpus_per_node=8, 
			cpus_per_task=96, 
			tasks_per_node=1, 
			timeout_min=self.job_ttl,
			job_name='maui_climber',
			account='maui',
			qos='maui_high',
			working_dir=self.workspace.resolve_path(version=version)
		)

		slurm_utils.JobObserver.shared.observe(
			job.id,
			callback=lambda res: self.set_results_for_version(version, res),
		)

		# Wait for current experiment and callbacks to finish
		await slurm.utils.JobObserver.shared.wait()

	def set_results_for_version(self, version: str, job_results: slurm_utils.JobResult):
		log_out = slurm_utils.get_logs_out(job_results.job_id, n=1)
		log_err = slurm_utils.get_logs_err(job_results.job_id, n=1)
		summary_response = self.run_scientist(
			prompts.SUMMARIZE_EXPERIMENT_LOGS.format(log_out, log_err),
			validator=lambda x: validators.validate_json(x, dict={})
		)
		outcome_summary = json.loads(summary_response)['summary']

		# Parse metrics from log file
		metrics = {}
		exp_logs_path = ... # @todo: find logs/<hash>.txt
		with open(exp_logs_path, 'r') as f:
			# @todo: Read file line by line and get last line with metrics
			pass

		if not metrics:
			metric_types = {k: type(v) for k, v in results.get('metrics', {}).items()}
			metrics_response = self.run_scientist(
				prompts.SUMMARIZE_EXPERIMENT_LOGS.format(log_out, log_err),
				validator=lambda x: validators.validate_json(x, metric_types)
			)
			metrics = json.loads(metrics_response)

		# In the worst case, default to empty metrics with previous keys
		if not metrics:
			metrics = {k: None for k, _ in results.get('metrics', {}).items()}

		res = {
			'status': job_results.status
			'metrics': metrics,
			'hypothesis': job_results.metadata['hypothesis'],
			'outcome_summary': outcome_summary
		}

	async def run(self, n_iterations=1):
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
