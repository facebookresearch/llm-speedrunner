GENERATE_CODE_HYPOTHESIS = """Study the current version of {fname}:

{code}

Then, consider the summary of this implementation and the result of running it. 
In the summary, the "hypothesis" value refers to the original hypothesis motivating this existing implementation.

{summary}

First, summarize at a high level what the current implementation does.
Next, come up with a new hypothesis for how you can improve the code to achieve the following:

{instruction}

Structure your response as a single JSON in the format below. Do not include any extra commentary in your final response.

{{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}}
"""