from typing import Optional


GENERATE_CODE_HYPOTHESIS = """Study the current code:

{code}

Then, consider the summary of this implementation and the result of running it. 
In the summary, the "hypothesis" value refers to the original hypothesis motivating this existing implementation.

{summary}

First, summarize at a high level what the current implementation does.
Then, come up with a new hypothesis for how you can improve the code to achieve the following:

{instruction}

Be methodical and scientific in suggesting changes. Avoid suggesting too many conceptual changes to the code at once, though some individual ideas may require more lines of code change, which is okay. Remember: Quality over quantity.

Structure your response as a single JSON in the format below. Do not include any extra commentary in your final response.

{{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}}
"""


HISTORY_INFO_COMPONENT = """To help in this task, consider this list of previous changes you have attempted along with their outcomes.

{history}
"""


def basic_ideation_prompt(
	code: str,
	summary: str, 
	instruction: str,
	history: Optional[str] = None,
):
	instructions = [instruction]

	if history:
		instructions.append(
			HISTORY_INFO_COMPONENT.format(history=history)
		)

	full_instructions = '\n'.join(instructions)

	return GENERATE_CODE_HYPOTHESIS.format(
		code=code,
		summary=summary,
		instruction=full_instructions,
	)
