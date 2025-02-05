from typing import Optional, Union
from pathlib import Path
import logging
import os

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from aider.llm import litellm

from core.agent import Agent
from core.prompts import coder_prompts
from core.workspace import Workspace


class AiderCoder(Agent):
    def __init__(
        self,
        model_url: Optional[str] = None,  # Should be "openai/<HF model id>"
        system_prompt: Optional[str] = None,
        log_llm_metrics=False,
        model_name: Optional[str] = None,
        stream=False,
        edit_format='diff',
        max_reflections=5,
        use_temperature: Union[bool, float] = False
    ):
        if system_prompt:
            logging.info('Currently, system prompt for AiderCoder is ignored.')

        chat_history_file = None
        if log_llm_metrics:
            chat_history_file = 'aider.txt'
        io = InputOutput(yes=True, chat_history_file=chat_history_file)

        os.environ['OPENAI_API_BASE'] = model_url
        os.environ['OPENAI_API_KEY'] = "sk-123"  # dummy value
        
        main_model = Model(model_name)
        if use_temperature is not None:
            main_model.use_temperature = use_temperature

        self._coder = Coder.create(
            main_model=main_model,
            io=io,
            stream=stream,
            use_git=False,
            edit_format=edit_format,
            summarize_from_coder=True,
        )
        self._coder.max_reflections = max_reflections

    def code(
        self, 
        instruction: str,
        fnames: str | list[str],
        workspace: Workspace,
        version: int,
        bug_history: Optional[str] = None,
        max_retries=1
    ) -> str:
        # Update history file
        aider_txt_path = workspace.resolve_path('aider.txt', version=version)
        self._coder.io.chat_history_file = Path(aider_txt_path)

        # Add code paths
        abs_fnames = [
            workspace.resolve_path(fname, version=version)
            for fname in fnames
        ]

        self._coder.abs_fnames.clear()
        for fname in abs_fnames:
            self._coder.abs_fnames.add(fname)

        code_prompt = coder_prompts.basic_code_prompt(
            fnames=fnames,
            instruction=instruction,
            packages=workspace.packages,
            bug_history=bug_history
        )

        coder_out = self._coder.run(code_prompt)

        if self._coder.summarizer_thread:
            self._coder.summarizer_thread.join()

        return coder_out

    def flush_logs(self, path: str):
        pass
