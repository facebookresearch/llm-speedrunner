"""
A basic ideator agent. 

Takes an instruction and produces a whole code edit, which can be saved.
"""
from typing import Optional
import json

from core.agent import Agent
from core.workspace import Workspace
from core import validators
from core.prompts import ideator_prompts


class Ideator(Agent):
	def ideate(
		self, 
		instruction: str,
		fnames: list[str],
		workspace: Workspace,
		version: int,
		n_ideas=1,
		max_retries=1
	) -> tuple[list[str], Optional[dict[str, str]]]:
		fname = None
		if len(fnames) > 1:
			raise ValueError('The base Ideator only supports a single fname.')
		else:
			fname = fnames[0]

		code = workspace.view(fname, version=version)
		summary = workspace.view('results.json', version=version)

		ideation_prompt = ideator_prompts.GENERATE_CODE_HYPOTHESIS.format(
			fname=fname,
			code=code,
			summary=summary,
			instruction=instruction
		)

		res_dict = json.loads(self.act(
			ideation_prompt,
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
			max_retries=max_retries
		))

		hypothesis = res_dict['hypothesis']

		return [hypothesis], {'summary': res_dict['summary']}
