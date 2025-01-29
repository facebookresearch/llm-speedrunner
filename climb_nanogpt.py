from datetime import datetime
from typing import Optional, Type, Union
import asyncio
import dataclasses
import logging
import json
import os
import re
import shutil
import subprocess
import sys
import time

from core.types import ExperimentConfig
from core.science_runner import ScienceRunner
from core.workspace import Workspace
from core.agent import Agent
from core import validators
from utils import fs_utils
from utils import slurm_utils
import prompts_nanogpt as prompts


NANOGPT_ENV_VARS = {
	'NANOGPT_TRAIN_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_train_*.bin',
	'NANOGPT_VAL_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_val_*.bin',
	'NANOGPT_VAL_TOKENS': 10485760
}

ENTRY_FILENAME = 'train_gpt.py'

MAX_LOG_LEN = 3000


class NanoGPTClimber(ScienceRunner):
	async def _run_exp(self, version: str):
		# See current solution
		code = self.workspace.view(ENTRY_FILENAME, version=version)
		summary = self.workspace.view('results.json', version=version)

		# Request next hypothesis
		hypothesis_res = self.run_scientist(
			prompts.GENERATE_HYPOTHESIS.format(code=code, summary=summary),
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
		)
		print(f'hypothesis_res:\n{hypothesis_res}')
		hypothesis = json.loads(hypothesis_res)['hypothesis']

		print(f'Hypothesis:\n{hypothesis}')

		# Implement hypothesis
		updated_code = self.run_scientist(
			prompts.IMPLEMENT_HYPOTHESIS.format(
				code=code,
				hypothesis=hypothesis
			),
			validator=validators.validate_code,odel
		)
		print(f'Updated code:\n{updated_code}', flush=True)

		# Save code to workspace's current version dir
		self.workspace.save_to_file(updated_code, ENTRY_FILENAME, version=version)

		# Send experiment to slurm
		job = slurm_utils.submit_job(
			command=ENTRY_FILENAME, 
			nodes=1, 
			tasks_per_node=8,
			gpus_per_node=8, 
			cpus_per_task=12,
			job_ttl=self.job_ttl,
			job_name='nanogpt',
			account='maui',
			working_dir=self.workspace.resolve_path(version=version),
			env_vars=NANOGPT_ENV_VARS,
		)

		# Monitor experiment status and bookkeep final outcome
		slurm_utils.JobObserver.shared.observe(
			job=job,
			metadata={'hypothesis': hypothesis},
			callback=lambda res: self.set_results_for_version(version, res),
		)

		# Wait for current experiment and callbacks to finish
		await slurm_utils.JobObserver.shared.wait()

		self.scientist.flush_logs(self.workspace.resolve_path('llm_history.jsonl', version=version))

	def set_results_for_version(self, version: str, job_results: slurm_utils.JobResult):
		log_out = job_results.log_out[0][-MAX_LOG_LEN:]
		log_err = job_results.log_err[0][-MAX_LOG_LEN:]
		outcome_summary = self.run_scientist(
			prompts.SUMMARIZE_LOGS_PROMPT.format(log_out=log_out, log_err=log_err)
		)
		print(f'outcome_summary:\n{outcome_summary}')

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
			summary = json.loads(
				self.workspace.view('results.json', version=version, no_filename_headers=True).strip()
			)
			metric_types = {k: Union[type(v), None] for k, v in summary.get('metrics', {}).items()}
			metric_types_str = json.dumps({k: type(v).__name__ for k, v in summary.get('metrics', {}).items()})
			metrics_response = self.run_scientist(
				prompts.PARSE_METRICS_FROM_LOGS.format(logs=log_out, metric_types=metric_types_str),
				validator=lambda x: validators.validate_json(x, metric_types)
			)
			print(f'metrics_response:\n{metrics_response}')
			if metrics_response:
				metrics = json.loads(metrics_response)

		# In the worst case, default to empty metrics with previous keys
		if not metrics:
			metrics = {k: None for k, _ in summary.get('metrics', {}).items()}

		job_results = {
			'status': job_results.status.value,
			'metrics': metrics,
			'hypothesis': job_results.metadata['hypothesis'],
			'outcome_summary': outcome_summary
		}

		self.workspace.save_to_file(json.dumps(job_results), 'results.json', version=version)


	async def run(self, n_iterations=5):
		for i in range(n_iterations):
			if i > 0:
				prev_version = str(i)
				version = self.workspace.create_version(from_version=prev_version)
			else:
				version = '1'
			await self._run_exp(version=version)


async def main():
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

	root_path = 'workspaces/nanogpt' + f'_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}'
	template_dir = 'workspace_templates/nanogpt'
	workspace = Workspace(root_path=root_path, template_dir=template_dir)

	exp_config = ExperimentConfig(
		preamble=prompts.TASK_PREAMBLE,
		job_ttl=1*60  # 1 hour
	)

	climber = NanoGPTClimber(config=exp_config, workspace=workspace, scientist=scientist)

	await climber.run(n_iterations=10)


if __name__ == '__main__':
	asyncio.run(main())
