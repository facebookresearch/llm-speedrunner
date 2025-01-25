from typing import Callable, Optional

from scientist.core.llm_client import LLMClient


class Agent:
	def __init__(self, model="gpt-4o", system_prompt: str):
		LLMClient(model=model, system_prompt=system_prompt)

	def act(self, instruction: str, validator: Optional[Callable[str, bool]], max_retries=1) -> str:
		response = LLMClient.generate(instruction)

		if validator and not validator(response):
			n_retries = 0
			while n_retries < max_retries:
				response = LLMClient.generate(instruction)

				n_retries += 1

				if validator(response):
					return response
		else:
			return response

		raise ValueError(f'Malformed response after {max_retries} attempts.')