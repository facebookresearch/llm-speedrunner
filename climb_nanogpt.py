from datetime import datetime
from typing import Optional, Type, Union
import asyncio
import dataclasses
import logging
import json
import os
import shutil
import subprocess
import sys
import time

from core.types import ExperimentConfig
from core.experiment_runner import ExperimentRunner
from core.workspace import Workspace
from core.agent import Agent
from core import validators
from utils import fs_utils
from utils import slurm_utils
from utils import code_utils
import prompts


NANOGPT_ENV_VARS = {
	'NANOGPT_TRAIN_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_train_*.bin',
	'NANOGPT_VAL_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_val_*.bin',
	'NANOGPT_VAL_TOKENS': 10485760
}

ENTRY_FILENAME = 'collatz.py'

MAX_LOG_LEN = 3000


class NanoGPTClimber(ExperimentRunner):
	async def _run_exp(self, version: str):
		# See current solution
		code = self.workspace.view(ENTRY_FILENAME, version=version)
		summary = json.loads(self.workspace.view('results.json', version=version))

		# Request next hypothesis
		hypothesis_res = self.run_scientist(
			prompts.NANOGPT_TASK_GENERATE_HYPOTHESIS.format(code=code, summary=summary),
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
		)
		hypothesis = json.loads(hypothesis_res)['hypothesis']

		# Implement hypothesis
		updated_code_response = self.run_scientist(
			prompts.NANOGPT_TASK_IMPLEMENT_HYPOTHESIS.format(
				code=code, 
				hypothesis=hypothesis,
				train_files=TRAIN_FILES,
				val_files=VAL_FILES,
				val_tokens=VAL_TOKENS,
			)
			validator=validators.validate_code,
		)
		updated_code = code_utils.extract_code(updated_code_response, strict=False)

		# Save code to workspace's current version dir
		self.workspace.save_to_file(updated_code, ENTRY_FILENAME, version=version)

		# Send experiment to slurm
		# job = slurm_utils.submit_job(
		# 	command=ENTRY_FILENAME, 
		# 	nodes=1, 
		# 	tasks_per_node=8,
		# 	gpus_per_node=8, 
		# 	cpus_per_task=12,
		# 	timeout_min=self.job_ttl,
		# 	job_name='nanogpt',
		# 	account='maui',
		# 	working_dir=self.workspace.resolve_path(version=version),
		# 	env_vars=NANOGPT_ENV_VARS,
		# )
		job = slurm_utils.submit_job(
			command=ENTRY_FILENAME, 
			nodes=1, 
			tasks_per_node=1,
			gpus_per_node=0, 
			cpus_per_task=12,
			timeout_min=self.job_ttl,
			job_name='test',
			account='maui',
			working_dir=self.workspace.resolve_path(version=version),
		)

		# Monitor experiment status and bookkeep final outcome
		slurm_utils.JobObserver.shared.observe(
			job=job.id,
			metadata={'hypothesis': hypothesis},
			log_dir=os.path.join(self.workspace.resolve_path(version=version), 'submitit_logs'),
			callback=lambda res: self.set_results_for_version(version, res),
		)

		# Wait for current experiment and callbacks to finish
		await slurm.utils.JobObserver.shared.wait()

		self.scientist.flush_logs(self.workspace.resolve_path('llm_history.jsonl', version=version))

	def set_results_for_version(self, version: str, job_results: slurm_utils.JobResult):
		log_out = job_results.log_out[0][-MAX_LOG_LEN:]
		log_err = job_results.log_err[0][-MAX_LOG_LEN:]
		outcome_summary = self.run_scientist(
			prompts.SUMMARIZE_LOGS_PROMPT.format(log_out, log_err)
		)

		# Parse metrics from log file
		metrics = {}
		matches = re.findall(r"step:(\d+)(?:/\d+)?\s+val_loss:([\d.]+)\s+train_time:(\d+)ms", log_out)
		if matches:
		    # Take the last match
		    last_match = matches[-1]
		    metrics = {
		        "n_steps": int(last_match[0]),
		        "val_loss": float(last_match[1]),
		        "train_time": int(last_match[2])
		    }

		if not metrics:
			metric_types = {k: Union[type(v), None] for k, v in results.get('metrics', {}).items()}
			metrics_response = self.run_scientist(
				prompts.PARSE_METRICS.format(text=log_out, metric_types=json.dumps(metric_types)),
				validator=lambda x: validators.validate_json(x, metric_types)
			)
			metrics = json.loads(metrics_response)

		# In the worst case, default to empty metrics with previous keys
		if not metrics:
			metrics = {k: None for k, _ in results.get('metrics', {}).items()}

		job_results = {
			'status': job_results.status.value
			'metrics': metrics,
			'hypothesis': job_results.metadata['hypothesis'],
			'outcome_summary': outcome_summary
		}

		self.workspace.save_to_file(json.dumps(job_results), 'results.json', version=version)


	async def run(self, n_iterations=1):
		for i in range(n_iterations):
			if i > 0:
				prev_version = str(i)
				version = self.workspace.create_version(from_version=prev_version)
			else:
				version = '1'
			await self._run_exp(version=version)


async def main():
	# Create scientist agent
	# node_id = "cr1-h100-p548xlarge-267"
	if len(sys.argv) != 2:
	    print("Usage: python climb_nanogpt.py <vllm server node_id>")
	    sys.exit(1)

	node_id = sys.argv[1]
	model_url = f"http://{node_id}.fair-aws-h100-2.hpcaas:8000/v1"
	scientist = Agent(
		model_url=model_url, 
		system_prompt=prompts.SCIENTIST_SYSTEM_PROMPT,
		log_llm_metrics=True
	)

	# root_path = 'workspaces/nanogpt' + f'_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}'
	# template_dir = 'workspace_templates/nanogpt'
	root_path = 'workspaces/collatz' + f'_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}'
	template_dir = 'workspace_templates/collatz'
	workspace = Workspace(root_path=root_path, template_dir=template_dir)

	exp_config = ExperimentConfig(
		preamble=prompts.NANOGPT_TASK_PREAMBLE,
		# job_ttl=1*60  # 1 hour
		job_ttl=2 # 2 minutes
	)

	climber = NanoGPTClimber(config=exp_config, workspace=workspace, scientist=scientist)

	await climber.run(n_iterations=10)


if __name__ == '__main__':
	asyncio.run(main())
