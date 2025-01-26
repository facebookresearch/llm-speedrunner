from typing import Optional
import json
import re

from openai import OpenAI


node_id = "cr1-h100-p548xlarge-267"
server_url = f"http://{node_id}.fair-aws-h100-2.hpcaas:8000/v1"


def remove_think_tokens(text: str):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


class LLMClient:
    def __init__(self, model_url: str):
        """LLM client to interface with VLLM and other LLM servers based on the OpenAI API.

        Args:
            model_url (str): url to model server
        """
        self.model_url = model_url
        self._client = OpenAI(
            base_url=server_url,
            api_key="token-abc123",
        )

    def generate(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        show_thinking=False
    ) -> str:
        """Query model_url for a chat completion.
        
        Args:
            system_prompt (str): A system prompt applied for all generations
            show_thinking (bool): If true, preserves any leading <thinking>...</thinking> tokens.

        """
        if system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
          model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
          messages=messages
        )
        # completion = json.dumps(dict(hypothesis='new hypothesis'))

        if not self.show_thinking:
            completion = strip_thinking_tokens(completion).strip()

        return completion
