from typing import Callable, Optional

from .llm_client import LLMClient


class Agent:
    def __init__(
        self,
        model_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        log_llm_metrics=False):
        self.llm = LLMClient(model_url=model_url, log_metrics=log_llm_metrics)

    def act(self, instruction: str, validator: Optional[Callable[str, Optional[str]]], max_retries=1) -> str:
        response = self.llm.generate(instruction)

        if validator:
            response = validator(response)

            n_retries = 0
            while n_retries < max_retries and response is None:
                response = LLMClient.generate(instruction, system_prompt=system_prompt)

                n_retries += 1

                response = validator(response)

        if response is None:
            raise ValueError(f'Malformed response after {max_retries} attempts.')

        return response

    def flush_logs(self, path: str):
        self.llm.flush_logs(path)