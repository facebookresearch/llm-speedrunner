# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import os
from .llm_client import LLMClient


class Agent:
    def __init__(
        self,
        model_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        log_llm_metrics=False,
        secrets: Optional[dict[str: str]] = None,
        api_version: Optional[str] = None,
    ):
        api_key = None
        if secrets:
            for k, v in secrets.items():
                if k.endswith('OPENAI_API_KEY') and 'gemini' not in model_name:
                    api_key = v
                    break
                if 'gemini' in model_name and k.endswith('GEMINI_API_KEY'):
                    api_key = v
                    os.environ['GEMINI_API_KEY'] = api_key
                    break


        self.llm = LLMClient(
            model_url=model_url,
            model_name=model_name,
            log_metrics=log_llm_metrics,
            api_key=api_key,
            api_version=api_version
        )
        self.system_prompt = system_prompt

    def act(
        self, 
        instruction: str,
        validator: Optional[Callable[str, Optional[str]]] = None, # type: ignore
        max_retries=1
    ) -> str:
        response = self.llm.generate(instruction)

        if validator:
            response = validator(response)

            n_retries = 0
            while n_retries < max_retries and response is None:
                response = self.llm.generate(instruction, system_prompt=self.system_prompt)

                n_retries += 1

                response = validator(response)

        if response is None:
            raise ValueError(f'Malformed response after {max_retries} attempts.')

        return response

    def flush_logs(self, path: str):
        self.llm.flush_logs(path)