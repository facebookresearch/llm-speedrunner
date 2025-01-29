from typing import Callable, Optional
import logging

from core.types import ExperimentConfig
from core.agent import Agent
from core.workspace import Workspace

class ScienceRunner:
	def __init__(self, config: ExperimentConfig, workspace: Workspace, scientist: Agent):
		self.preamble = config.preamble
		self.max_retries = config.max_retries
		self.job_ttl = config.job_ttl
		self.workspace = workspace
		self.scientist = scientist

	def get_instruction(self, instruction: str) -> str:
		return '\n'.join([self.preamble, instruction])

	def run_scientist(
		self, 
		instruction: str, 
		validator: Optional[Callable[str, bool]] = None, 
		max_retries: Optional[int] = None
	) -> str:
		if max_retries is None:
			max_retries = self.max_retries

		try:
			response = self.scientist.act(
				self.get_instruction(instruction),
				validator=validator,
				max_retries=max_retries
			)
		except:
			logging.info("Bad response. Terminating experiment prematurely.")
			response = None

		return response

	async def run(self, n_iterations=1):
		raise NotImplementedError()
