from typing import Optional, Union
import asyncio
import json
import re
import sys

from openai import OpenAI, AzureOpenAI


def strip_think_tokens(text: str):
    # return re.sub(r"<think>\n.*?\n</think>", "", text, flags=re.DOTALL)
    if '</think>' in text:
        return text.split('</think>\n\n')[1]


def get_model_client(
    model_url: str, 
    api_key: str,
    api_version='2024-10-21', 
    timeout=30*60
) -> Union[OpenAI, AzureOpenAI]:
    if 'https://azure' in model_url:
        client = AzureOpenAI(
          api_key=api_key,  
          azure_endpoint=model_url,
          api_version=api_version
        )
    else:
        client = OpenAI(
            base_url=model_url,
            api_key=api_key,
            timeout=30*60, # 30 min timeout since we are <thinking/>
        )

    return client


class LLMClient:
    def __init__(
        self,
        model_url: str,
        model_name: str,
        log_metrics=False,
        api_key: str = 'token-abc123'
    ):
        """LLM client to interface with VLLM and other LLM servers based on the OpenAI API.

        Args:
            model_url (str): url to model server
        """
        self.model_url = model_url
        self.model_name = model_name
        self._client = get_model_client(model_url=model_url, api_key=api_key)

        self._log = []
        self._log_metrics = log_metrics

    @property
    def is_system_prompt_enabled(self):
        if self.model_name == 'o1-preview':
            return False
        else:
            return True

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
        print('PROMPT:\n', prompt)
        if system_prompt and self.is_system_prompt_enabled:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        
        completion = self._client.chat.completions.create(
          model=self.model_name,
          messages=messages
        )

        res_content = completion.choices[0].message.content

        if not show_thinking:
            final_res = strip_think_tokens(res_content).strip()
        else:
            final_res = res_content

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
    if len(sys.argv) != 4:
        print("Usage: python llm_client.py <model_url> <model_name> <api_key>")
        sys.exit(1)

    model_url = sys.argv[1]
    print(f'model_url={model_url}')

    api_key = sys.argv[3]

    model_name = sys.argv[2]
    llm = LLMClient(model_url=model_url, model_name=model_name, log_metrics=True, api_key=api_key)


    # for _ in range(10):
    while True:
        prompt = input("Enter prompt: ")
        res = llm.generate(
            prompt, 
            # system_prompt='Respond as if you are Shrek.', 
            show_thinking=True
        )

        print("RESPONSE:\n", res)
        # print(res)

    # llm.flush_logs('journal.jsonl')

