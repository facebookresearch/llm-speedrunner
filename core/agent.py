from typing import Callable, Optional

from .llm_client import LLMClient


class Agent:
	def __init__(
		self,
		model="gpt-4o",
		system_prompt: Optional[str] = None,
		log_llm_metrics=False):
		self.llm = LLMClient(model=model, log_metrics=log_llm_metrics)

	def act(self, instruction: str, validator: Optional[Callable[str, bool]], max_retries=1) -> str:
		response = self.llm.generate(instruction)

		if validator and not validator(response):
			n_retries = 0
			while n_retries < max_retries:
				response = LLMClient.generate(instruction, system_prompt=system_prompt)

				n_retries += 1

				if validator(response):
					return response
		else:
			return response

		raise ValueError(f'Malformed response after {max_retries} attempts.')

	def flush_logs(self, path: str):
		self.llm.flush_logs(path)