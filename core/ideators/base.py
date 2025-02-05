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
		history: Optional[str] = None,
		max_retries=1
	) -> tuple[list[str], Optional[dict[str, str]]]:
		abs_paths = [workspace.resolve_path(x, version=version) for x in fnames]
		code = workspace.view(abs_paths, version=version)
		summary = workspace.view('results.json', version=version)

		ideation_prompt = ideator_prompts.basic_ideation_prompt(
			code=code,
			summary=summary,
			instruction=instruction,
			history=history
		)

		res_dict = json.loads(self.act(
			ideation_prompt,
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
			max_retries=max_retries
		))

		hypothesis = res_dict['hypothesis']

		return [hypothesis], {'summary': res_dict['summary']}
