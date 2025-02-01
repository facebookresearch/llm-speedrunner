"""
A basic coder agent. 

Takes an instruction and produces a whole code edit, which can be saved.
"""
from core.agent import Agent
from core.workspace import Workspace
from core import validators
from core.prompts import coder_prompts


class Coder(Agent):
	def code(
		self, 
		instruction: str,
		fnames: list[str],
		workspace: Workspace,
		version: int,
		max_retries=1
	) -> str:
		fname = None
		if len(fnames) > 1:
			raise ValueError('The base Coder only supports a single fname.')
		else:
			fname = fnames[0]

		code = workspace.view(fname, version=version)
				
		update_prompt = coder_prompts.UPDATE_SINGLE_FILE.format(
			fname=fname,
			instruction=instruction,
			code=code
		)

		updated_code = self.act(
			update_prompt,
			validator=validators.validate_code,
			max_retries=max_retries
		)

		workspace.save_to_file(updated_code, fname, version=version)

		return updated_code