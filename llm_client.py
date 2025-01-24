import asyncio


class LLMClient:
	def __init__(model='qwen-r1-32b', system_prompt=Optional[str] = None):
		self.model = model
		self.system_prompt = system_prompt

	def generate(self, x: str) -> str:
		return "test"