import asyncio

from openai import OpenAI


node_id = "cr1-h100-p548xlarge-267"
server_url = f"http://{node_id}.fair-aws-h100-2.hpcaas:8000/v1"


# @todo: Support asyncio
# @todo: Support <thinking/>
class LLMClient:
	def __init__(model='qwen-r1-32b', system_prompt=Optional[str] = None):
		self.model = model
		self.system_prompt = system_prompt

		self._client = OpenAI(
		    base_url=server_url,
		    api_key="token-abc123",
		)

	def generate(self, prompt: str, max_tokens=None) -> str:
		completion = client.chat.completions.create(
		  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
		  messages=[
		    {"role": "user", "content": prompt}
		  ]
		)
