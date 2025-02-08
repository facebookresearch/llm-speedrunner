from typing import Optional


BASIC_CODE_PREAMBLE = """Study the current version of {fnames}:
"""


CHILD_BUG_INFO_COMPONENT = """To help with your task, here is a list summarizing recent erroneous changes to the above code that you have previously tried, along with a summary of the outcome of each change.
{history}
"""


PACKAGE_INFO_COMPONENT= """Avoid installing any additional packages. Assume you have access to the following packages outside of the standard python packages:
{packages}
"""


BASIC_CODE_PROMPT = """Your goal is to implement the following ideas for improving the code:

{ideas}

{instruction}

I trust you to make good decisions, so do not ask me for permission to make any code changes.

In your final response, include ONLY the fully-functional updated code which implements ideas in the hypothesis above. Do NOT include any other content in your final response besides the code.
"""


def basic_code_prompt(
	fnames: list[str], 
	instruction: str,
	ideas: Optional[str] = None,
	code: Optional[str] = None,
	packages: Optional[list[str]] = None,
	bug_history: Optional[str] = None,
):
	if len(fnames) == 1:
		fnames = fnames[0]
	else:
		fnames = '\n'.join([f'- {x}' for x in fnames])
	preamble = BASIC_CODE_PREAMBLE.format(fnames=fnames)

	if code:
		preamble = preamble + '\n' + code + '\n'

	instructions = [instruction + '\n']
	if packages:
		package_list = '\n'.join([f'- {x}' for x in packages])
		instructions.append(
			PACKAGE_INFO_COMPONENT.format(packages=package_list)
		)
	if bug_history:
		instructions.append(
			CHILD_BUG_INFO_COMPONENT.format(history=bug_history)
		)

	return preamble + '\n' + BASIC_CODE_PROMPT.format(
		ideas=ideas,
		instruction='\n'.join(instructions).rstrip()
	)
