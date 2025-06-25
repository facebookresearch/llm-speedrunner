# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import asyncio
import json
import re
import sys
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI, AzureOpenAI
import litellm


def strip_think_tokens(text: str):
    if '</think>' in text:
        return re.sub(r"(?:<think>)?.*?</think>", "", text, flags=re.DOTALL)
    return text


def get_model_client(
    model_url: str, 
    model_name: str,
    api_key: str,
    api_version='2024-12-01-preview', 
    timeout=30*60
) -> Union[OpenAI, AzureOpenAI]:
    if 'gemini' in model_name:
        client = lambda prompt: litellm.completion(
            model=model_name,
            messages=prompt,
            api_key=api_key,
            reasoning_effort='high'
        )
    elif 'https://azure' in model_url:
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
        api_key: str = 'token-abc123',
        max_retries: int = 3,
        api_version: str = '2024-10-21'
    ):
        """LLM client to interface with VLLM and other LLM servers based on the OpenAI API.

        Args:
            model_url (str): url to model server
            model_name (str): name of the model to use
            log_metrics (bool): whether to log metrics
            api_key (str): API key for authentication
            max_retries (int): maximum number of retry attempts for API calls
        """
        # import ipdb; ipdb.set_trace()
        self.model_url = model_url
        if 'gemini' in model_name:
            model_name = 'gemini/gemini-2.5-flash-preview-04-17'
        self.model_name = model_name
        # self._client = get_model_client(model_url=model_url, api_key=api_key, api_version=api_version)
        self._client = get_model_client(model_url=model_url, model_name=model_name, api_key=api_key, )

        self.max_retries = max_retries

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_api_call(self, messages):
        """Make the actual API call with retry logic."""
        if 'gemini' in self.model_name:
            return self._client(messages)
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages
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

        Returns:
            str: The generated response from the model
        """
        print('PROMPT:\n', prompt)
        if system_prompt and self.is_system_prompt_enabled:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        
        try:
            completion = self._make_api_call(messages)
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            raise

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
