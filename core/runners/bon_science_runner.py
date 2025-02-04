from typing import Callable, Optional
import dataclasses
import logging

import numpy as np

from utils import slurm_utils
from core.types import ExperimentConfig, SlurmConfig
from core.agent import Agent
from core.coders.base import Coder
from core.ideators.base import Ideator
from core.runners.science_runner import ScienceRunner
from core.workspace import Workspace

import random
import json


class BoNScienceRunner(ScienceRunner):
	"""Science loop that evaluates N hypotheses, and continues with the BoN."""
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
		max_log_len=30_000,
		n_hypotheses: int = 1,
	):
		super().__init__(
			config=config,
			workspace=workspace,
			assistant=assistant,
			ideator=ideator,
			coder=coder,
			slurm_config=slurm_config,
			eval_slurm_config=eval_slurm_config,
			max_retries=max_retries,
			max_log_len=max_log_len
		)

		self.n_hypotheses = n_hypotheses

	def _job_callback(self, version: str, job_results: slurm_utils.JobResult):
		self.set_results_for_version(version, job_results)

		self.assistant.flush_logs(
			self.workspace.resolve_path('llm_history.jsonl', 
			version=version)
		)

	async def _run_exp(
		self,
		version: str, 
		metadata: Optional[dict[str, str | int | float, bool]] = None
	):
		coder_out = self.coder.code(
			instruction=self.code_instructions,
			fnames=self.fnames,
			workspace=self.workspace,
			version=version,
			max_retries=self.max_retries
		)
		print(f'Coder out:\n{coder_out}')

		# Send experiment to slurm
		if self.slurm_config.use_torchrun:
			command = self.entry_fname
		else:
			command = f'python {self.entry_fname}'

		job = slurm_utils.submit_job(
			command=command, 
			working_dir=self.workspace.resolve_path(version=version),
			**dataclasses.asdict(self.slurm_config)
		)

		# Monitor experiment status and bookkeep final outcome
		callback = None
		if self.eval_fname is None:
			callback = lambda res: self._job_callback(version, res)

		slurm_utils.JobObserver.shared.observe(
			job=job,
			metadata=metadata,
			callback=callback,
		)

	async def _run_eval(
		self, 
		version: str,
		metadata: Optional[dict[str, str | int | float, bool]] = None
	):
		if self.eval_fname is not None:
			eval_slurm_config = self.eval_slurm_config
			if not eval_slurm_config:
				eval_slurm_config = self.slurm_config

			if self.eval_slurm_config.use_torchrun:
				command = self.eval_fname
			else:
				command = f'python {self.eval_fname}'

			eval_job = slurm_utils.submit_job(
				command=f"python {self.eval_fname}", 
				working_dir=self.workspace.resolve_path(version=version),
				**dataclasses.asdict(self.eval_slurm_config)
			)

			slurm_utils.JobObserver.shared.observe(
				job=job,
				metadata=metadata,
				callback=lambda res: self._job_callback(version, res)
			)

	async def run(self, n_iterations=1):
		open_version = None
		for i in range(n_iterations):
			# Request next hypotheses
			hypotheses, _ = self.ideator.ideate(
				instruction=self.idea_instructions,
				fnames=self.fnames,
				workspace=self.workspace,
				version='1' if not open_version else open_version,
				n_ideas=self.n_hypotheses,
				max_retries=1
			)

			current_versions = []
			if open_version is not None:
				current_versions.append(open_version)  # Always consider best so far

			version2metadata = {}
			for hyp_idx, hypothesis in enumerate(hypotheses):
				print(f'Hypothesis:\n{hypothesis}')

				# Create a new workspace version for each experiment
				if i == 0 and hyp_idx == 0:
					# Use initial workspace for first hypothesis
					version = '1'
				elif i == 0:
					# All first generation hypotheses branch from template
					version = self.workspace.create_version()
				else:
					# All other hypotheses branch from best version so far
					prev_version = open_version
					version = self.workspace.create_version(
						from_version=prev_version
					)

				current_versions.append(str(version))
				version2metadata[version] = {
					'hypothesis': hypothesis
				}

				# Schedule experiments for all hypotheses
				await self._run_exp(
					version=version,
					metadata=version2metadata[version]
				)

			# Wait for all experiments to finish
			await slurm_utils.JobObserver.shared.wait()

			if self.eval_fname is not None:
				for version in current_versions:
					metadata = version2metadata[version]
					self._run_eval(version=version, metadata=metadata)

				await slurm_utils.JobObserver.shared.wait()

			# Set open set to the top-1 version
			open_version = self.workspace.get_top_k_versions(
				selection_metric=self.selection_metric,
				from_versions=current_versions,
				lower_is_better=self.lower_is_better,
				k=1
			)[0]
