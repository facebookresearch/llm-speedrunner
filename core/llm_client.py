from typing import Optional
import asyncio
import json
import re
import sys

from openai import OpenAI


def strip_think_tokens(text: str):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


class LLMClient:
    def __init__(self, model_url: str, log_metrics=False):
        """LLM client to interface with VLLM and other LLM servers based on the OpenAI API.

        Args:
            model_url (str): url to model server
        """
        self.model_url = model_url
        self._client = OpenAI(
            base_url=model_url,
            api_key="token-abc123",
            timeout=30*60, # 30 min timeout since we are <thinking/>
        )

        self._log = []
        self._log_metrics = log_metrics

    def flush_logs(self, path: str):
        if self._log_metrics:
            # Write log to log_url
            with open(path, 'a') as f:
                for entry in self._log:
                    f.write(json.dumps(entry) + '\n')

            self._log = []

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
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        
        completion = self._client.chat.completions.create(
          model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
          messages=messages
        )

        res_content = completion.choices[0].message.content

        if not show_thinking:
            final_res = strip_think_tokens(res_content).strip()

        if self._log_metrics:
            self._log.append(
                dict(
                    prompt=prompt, 
                    response=final_res, 
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                )
            )

        return final_res


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llm_client.py <vllm server node_id> <prompt>")
        sys.exit(1)

    node_id = sys.argv[1]
    model_url = f"http://{node_id}.fair-aws-h100-2.hpcaas:8000/v1"
    print(f'model_url={model_url}')

    llm = LLMClient(model_url=model_url, log_metrics=True)

    prompt = sys.argv[2]

    for _ in range(10):
        res = llm.generate(
            prompt, 
            system_prompt='Respond as if you are Shrek.', 
            show_thinking=False
        )

        print(res)

    llm.flush_logs('journal.jsonl')

