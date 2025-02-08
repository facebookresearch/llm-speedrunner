"""
A basic coder agent. 

Takes an instruction and produces a whole code edit, which can be saved.
"""
from typing import Optional

from core.agent import Agent
from core.workspace import Workspace
from core import validators
from core.prompts import coder_prompts


class Coder(Agent):
	def code(
		self, 
		instruction: str,
        ideas: str,
		fnames: list[str],
		workspace: Workspace,
		version: int,
		bug_history: Optional[str] = None,
		max_retries=1
	) -> str:
		abs_paths = workspace.resolve_path(fnames, version=version)
		code = workspace.view(abs_paths, version=version)
				
		update_prompt = coder_prompts.basic_code_prompt(
			fnames=fnames,
			instruction=instruction,
			ideas=ideas,
			code=code,
			packages=workspace.packges,
			bug_history=bug_history
		)

		updated_code = self.act(
			update_prompt,
			validator=validators.validate_code,
			max_retries=max_retries
		)

		workspace.save_to_file(updated_code, fname, version=version)

		return updated_code