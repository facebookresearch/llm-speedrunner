from typing import Callable, Optional

from .llm_client import LLMClient


class Agent:
    def __init__(
        self,
        model_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        log_llm_metrics=False,
        secrets: Optional[dict[str: str]] = None):

        api_key = None
        if secrets:
            for k, v in secrets.items():
                if k.endswith('OPENAI_API_KEY'):
                    api_key = v
                    break

        self.llm = LLMClient(
            model_url=model_url,
            model_name=model_name,
            log_metrics=log_llm_metrics,
            api_key=api_key
        )
        self.system_prompt = system_prompt

    def act(
        self, 
        instruction: str,
        validator: Optional[Callable[str, Optional[str]]] = None,
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