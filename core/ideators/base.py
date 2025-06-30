# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
		task_description: str,
		fnames: list[str],
		workspace: Workspace,
		version: int,
		ignore_ideas: Optional[str] = None,
		history: Optional[str] = None,
		knowledge: Optional[str] = None,
		max_retries=1
	) -> tuple[list[str], Optional[dict[str, str]]]:
		version_info = workspace.get_version_info(version)
		if version == '0':
			parent_version = version
		else:
			assert version_info.parent_version is not None, 'Version must have a parent'
			parent_version_info = workspace.get_version_info(version_info.parent_version)
			parent_version = parent_version_info.version

		# Generate new ideas based on the contents of the parent version
		abs_paths = [workspace.resolve_path(x, version=parent_version) for x in fnames]
		code = workspace.view(abs_paths, version=parent_version)
		summary = version_info.get_summary_string(with_version_headers=False)

		ideation_prompt = ideator_prompts.basic_ideation_prompt(
			code=code,
			summary=summary,
			task_description=task_description,
			is_debug=version_info.bug_depth > 0,
			ignore_ideas=ignore_ideas,
			history=history,
			knowledge=knowledge,
		)

		res_dict = json.loads(self.act(
			ideation_prompt,
			validator=lambda x: validators.validate_json(x, dict(hypothesis=str)),
			max_retries=max_retries
		))

		hypothesis = res_dict['hypothesis']

		return hypothesis, {'summary': res_dict['summary']}
