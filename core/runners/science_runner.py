from typing import Callable, Optional, Union
import dataclasses
import json
import logging

from utils import metrics_utils
from utils import slurm_utils
from utils import str_utils
from core.types import ExperimentConfig, SlurmConfig
from core.agent import Agent
from core.coders.base import Coder
from core.ideators.base import Ideator
from core.workspace import Workspace
from core.prompts import analysis_prompts
from core import validators


class ScienceRunner:
	def __init__(
		self, 
		config: ExperimentConfig, 
		workspace: Workspace, 
		assistant: Agent,
		ideator: Ideator,
		coder: Coder,
		slurm_config: SlurmConfig,
		eval_slurm_config: Optional[SlurmConfig] = None,
		max_retries=3,
		max_log_len=30_000
	):
		self.preamble = config.preamble
		self.idea_instructions = config.idea_instructions
		self.code_instructions = config.code_instructions
		self.fnames = config.fnames

		self.entry_fname = config.entry_fname
		self.eval_fname = config.eval_fname
		self.slurm_config = slurm_config
		self.eval_slurm_config = eval_slurm_config

		self.workspace = workspace

		self.selection_metric = config.selection_metric
		self.lower_is_better = config.lower_is_better
		self.metric_types = {
			k: str_utils.basic_type_name_to_type(v) 
			for k,v in config.metric_types.items()
		}
		self.metrics_at_least = config.metrics_at_least
		self.metrics_at_most = config.metrics_at_most
		self.max_retries = max_retries
		self.max_log_len = max_log_len

		# Agents
		self.assistant = assistant
		self.ideator = ideator
		self.coder = coder

	def get_instruction(self, instruction: str) -> str:
		return '\n'.join([self.preamble, instruction])

	def set_results_for_version(self, version: str, job_results: slurm_utils.JobResult):
		log_out = job_results.log_out[0][-self.max_log_len:]
		log_err = job_results.log_err[0][-self.max_log_len:]
		outcome_summary = self.assistant.act(
			analysis_prompts.SUMMARIZE_LOGS_PROMPT.format(log_out=log_out, log_err=log_err)
		)
		print(f'outcome_summary:\n{outcome_summary}')

		# Parse metrics from log file
		metrics = {}
		if self.metric_types is not None:
			metrics = metrics_utils.extract_best_line_metrics(
				log_out, 
				metric_types=self.metric_types,
				selection_metric=self.selection_metric,
				lower_is_better=self.lower_is_better,
				metrics_at_least=self.metrics_at_least,
				metrics_at_most=self.metrics_at_most
			)

		# If no regex match on results
		if not metrics:
			summary = json.loads(
				self.workspace.view('results.json', version=version, no_filename_headers=True).strip()
			)
			metric_types = {k: Union[type(v), None] for k, v in summary.get('metrics', {}).items()}
			metric_types_str = json.dumps({k: type(v).__name__ for k, v in summary.get('metrics', {}).items()})
			metrics_response = self.assistant.act(
				analysis_prompts.PARSE_METRICS_FROM_LOGS.format(logs=log_out, metric_types=metric_types_str),
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
			**job_results.metadata,
			'outcome_summary': outcome_summary
		}

		self.workspace.save_to_file(json.dumps(job_results), 'results.json', version=version)


	async def run(self, n_iterations=1):
		raise NotImplementedError()

