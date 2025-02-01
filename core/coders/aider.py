from typing import Optional
from pathlib import Path
import logging
import os

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from aider.llm import litellm

from core.agent import Agent
from core.workspace import Workspace


class AiderCoder(Agent):
    def __init__(
        self,
        model_url: Optional[str] = None,  # Should be "openai/<HF model id>"
        system_prompt: Optional[str] = None,
        log_llm_metrics=False,
        model_name: Optional[str] = None,
        stream=False,
        edit_format='diff'):

        if system_prompt:
            logging.info('Currently, system prompt for AiderCoder is ignored.')

        chat_history_file = None
        if log_llm_metrics:
            chat_history_file = 'aider.txt'
        io = InputOutput(yes=True, chat_history_file=chat_history_file)

        os.environ['OPENAI_API_BASE'] = model_url
        os.environ['OPENAI_API_KEY'] = "sk-123"  # dummy value
        
        main_model = Model(model_name)

        self._coder = Coder.create(
            main_model=main_model,
            io=io,
            stream=stream,
            use_git=False,
            edit_format=edit_format,
            summarize_from_coder=True
        )

    def code(
        self, 
        instruction: str,
        fnames: str | list[str],
        workspace: Workspace,
        version: int,
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

        coder_out = self._coder.run(instruction)

        if self._coder.summarizer_thread:
            self._coder.summarizer_thread.join()

        return coder_out
