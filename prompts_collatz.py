SCIENTIST_SYSTEM_PROMPT = """You are a machine learning scientist, with expertise in large language models and high-performance computing. Use your expertise to assist the user in their machine learning task.
"""

# General
SUMMARIZE_LOGS_PROMPT = """Task: Produce a succinct summary of the following stdout and stderr logs for a job running on a compute cluster. Keep your summary below 500 words.

# stdout logs
{log_out}


# stderr logs
{log_err}

Respond with just your summary text with no extra commentary and no extra formatting.
"""


PARSE_METRICS_FROM_LOGS = """Task: Analyze the following output logs and extract metrics following the metrics structure and typing template provided below. 

# Logs
{logs}

# Metric dict template (showing expected type for each key)
{metric_types}

Respond with only the extracted metrics as a JSON dict following the exact structure and type specification in the dict template below. 
If no metrics are successfully extracted, return the empty dict, {{}}. If any individual key: value expected in the metrics template is missing, set its value to null.
"""


# Collatz-specific prompts
TASK_PREAMBLE = """Your goal is to improve the logic in collatz.py to find the longest Collatz sequence with a runtime budget of 1 minute.

Your script will be terminated automatically when the time is up. You will be judged based on any results printed to stdout within this time.
"""

GENERATE_HYPOTHESIS = """Study the current version of collatz.py:

{code}

Then, consider the summary of this implementation and the result of running it. In the summary, the hypothesis value refers to the original hypothesis motivating this implementation.

{summary}

First, summarize at a high level what the current implementation does.
Next, come up with a new hypothesis for how you can improve this code to find the longest Collatz sequence within a runtime budget of 1 minute.

Structure your response as a single JSON in the format below. Do not include any extra commentary in your final response.

{{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}}
"""


IMPLEMENT_HYPOTHESIS = """Study the current version of collatz.py:

{code}

Now implement the following hypothesis for improving the code. 

{hypothesis}

Make sure you do not change the logging statements, so that the results continue to printed to stdout in the same format. Otherwise, the experiment run may be deemed invalid.
Besides the logging statements, you can change anything about the script, including the limit.

In your final response, include ONLY the fully-functional updated code which implements ideas in the hypothesis above. Do NOT include any other content in your final response besides the code.
"""
