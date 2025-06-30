# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


GENERATE_CODE_HYPOTHESIS = """Study the current code:

{code}

Then, consider the summary of this implementation and the result of running it. 
In the summary, the "hypothesis" value refers to the original hypothesis motivating this existing implementation.

{summary}

First, summarize at a high level what the current implementation does.
Then, come up with a new hypothesis for how you can improve the code to do as well as possible in the following task:

# Task description
{instruction}

# Idea guidelines
- Your idea will be handed to an expert ML engineer to implement. You must therefore be conceptually precise and ideally provide a concrete and detailed design of the implementation.
- The engineer only has 1 minute to read your idea and design spec, so be mindful to keep these descriptions as concise as possible.
- Your goal is to achieve the state-of-art in the task described. Be ambitious in ideation, so long as the solution adheres to any task constraints specified above.
"""


DEBUG_CODE_HYPOTHESIS = """Study the current code:

{code}

Consider the issues described in the following summary, which occur when running the code:

{summary}

First summarize at a high level what the current implementation does and why the bug might arise. 
Then come up with a hypothesis for how you can fix these issues with the code, while making sure that it solves the following task:

# Task description
{instruction}
"""


JSON_FORMAT_INSTRUCTION = """Structure your response as a single JSON in the format below. Do not include any extra commentary in your final response.

{{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}}
"""

IGNORE_IDEAS_INFO_COMPONENT = """In your ideation, ignore the following ideas, which have already been proposed:

{ideas}
"""


HISTORY_INFO_COMPONENT = """To help in this task, consider this list of previous changes you have attempted along with their outcomes.

{history}
"""


KNOWLEDGE_INFO_COMPONENT = """You may also wish to consider the following relevant information to inform your idea generation.

{knowledge}
"""


def basic_ideation_prompt(
	code: str,
	summary: str, 
	task_description: str,
	is_debug=False,
	ignore_ideas: Optional[list[str]] = None,
	history: Optional[str] = None,
	knowledge: Optional[str] = None,
):
	instructions = [task_description]

	if ignore_ideas:
		ignore_list = '\n'.join([f'<idea>{x}</idea>' for x in ignore_ideas])
		ignore_summary = f'<ignore_ideas>\n{ignore_list}\n</ignore_ideas>'
		instructions.append(
			IGNORE_IDEAS_INFO_COMPONENT.format(ideas=ignore_summary)
		)

	if history:
		instructions.append(
			HISTORY_INFO_COMPONENT.format(history=history)
		)

	if knowledge:
		instructions.append(
			KNOWLEDGE_INFO_COMPONENT.format(knowledge=knowledge)
		)

	full_instructions = '\n'.join(instructions) + '\n' + JSON_FORMAT_INSTRUCTION

	template = DEBUG_CODE_HYPOTHESIS if is_debug else GENERATE_CODE_HYPOTHESIS

	return template.format(
		code=code,
		summary=summary,
		instruction=full_instructions,
	)
