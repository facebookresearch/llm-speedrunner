# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


BASIC_CODE_PREAMBLE = """Study the current version of {fnames}:
"""


CHILD_BUG_INFO_COMPONENT = """To help with your task, here is a list summarizing recent erroneous changes to the above code that you have previously tried, along with a summary of the outcome of each change.
{history}
"""


PACKAGE_INFO_COMPONENT= """**Never** install or ask to install any additional packages. Assume you have access to the following packages outside of the standard python packages:
{packages}

If necessary, you may access pretrained model checkpoints via HuggingFace for smaller models like BERT variants or CLIP.
"""

KNOWLEDGE_INFO_COMPONENT = """You have access to the following knowledge, consider these when writing code:
{knowledge}
"""


BASIC_CODE_PROMPT = """Your goal is to implement the following ideas to improve the code so that it better achieves the task:

# Ideas
{ideas}

# Task description
{instruction}

I trust you to make good decisions, so do not ask me for permission to make any code changes. 
Do not ever ask to install any additional packages. The answer will be no.

In your final response, include ONLY the fully-functional updated code which implements ideas in the hypothesis above. Do NOT include any other content in your final response besides the code.
"""

ZERO_KNOWLEDGE_CODE_PROMPT = """Your goal is to improve the code to achieve the following task:

# Task description
{instruction}

First, analyze the task and come up with a plan for solving the task:
1. Consider ideas for changes and improvements needed to improve on the task. Consider both creative and practical ideas.
2. Break down the implementation into clear steps, generate pseudo codes for each step
3. Consider potential challenges and how to address them

Then, implement your plan by making the necessary code changes.

I trust you to make good decisions, so do not ask me for permission to make any code changes.
Do not ever ask to install any additional packages. The answer will be no.

Respond with your plan for improving the code, followed by the fully-functional updated code implementing your plan.
"""

STRICT_DIFF_PROMPT = """
You will edit the code using the diff format, when generating the diff, make sure the generated SEARCH block will **EXACTLY** match the code you will edit.
Do not skip any lines especially in the SEARCH block as missing anything will results in the code not being edited.
Do not change any indentation, the SEARCH block should have the same indentation as the code you will edit, otherwise the code will not be edited.
"""

def basic_code_prompt(
	task_description: str, 
	fnames: list[str],
	instruction: Optional[str],
	ideas: Optional[str],
	code: Optional[str] = None,
	packages: Optional[list[str]] = None,
	bug_history: Optional[str] = None,
	knowledge: Optional[str] = None
):
	if len(fnames) == 1:
		fnames = fnames[0]
	else:
		fnames = '\n'.join([f'- {x}' for x in fnames])
	preamble = BASIC_CODE_PREAMBLE.format(fnames=fnames)

	if code:
		preamble = preamble + '\n' + code + '\n'

	instructions = [task_description + '\n']
	if instruction:
		instructions.append(instruction + '\n')

	if knowledge:
		instructions.append(
			KNOWLEDGE_INFO_COMPONENT.format(knowledge=knowledge)
		)

	if packages:
		package_list = '\n'.join([f'- {x}' for x in packages])
		instructions.append(
			PACKAGE_INFO_COMPONENT.format(packages=package_list)
		)
	if bug_history:
		instructions.append(
			CHILD_BUG_INFO_COMPONENT.format(history=bug_history)
		)

	if not len(ideas) and not knowledge:
		# this case we use a dummy ideator and zero knowledge
		# ideas should be '', and knowledge should be None
		return preamble + '\n' + ZERO_KNOWLEDGE_CODE_PROMPT.format(
			instruction='\n'.join(instructions).rstrip()
		)

	return preamble + '\n' + BASIC_CODE_PROMPT.format(
		ideas=ideas,
		instruction='\n'.join(instructions).rstrip()
	)