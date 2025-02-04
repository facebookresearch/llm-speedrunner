UPDATE_SINGLE_FILE = """Study the current version of {fname}:

{code}

Now implement the following ideas for improving the code. 

{instruction}

I trust you to make good decisions, so do not ask me for permission to make any code changes.

Avoid installing any additional packages. Assume you have access to the following packages outside of the standard python packages:
- numpy
- numba
- torch

In your final response, include ONLY the fully-functional updated code which implements ideas in the hypothesis above. Do NOT include any other content in your final response besides the code.
"""