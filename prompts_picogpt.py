SCIENTIST_SYSTEM_PROMPT = """You are a machine learning scientist, with expertise in large language models and high-performance computing. Use your expertise to assist the user in their machine learning task.
"""

# General
SUMMARIZE_LOGS_PROMPT = """Task: Produce a succinct summary of the following stdout and stderr logs for a job running on a GPU cluster. Keep your summary below 500 words.

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


# NanoGPT speedrun prompts
TASK_PREAMBLE = """Your goal is to improve the model architecture and other training logic in main.py, so that running `python main.py` results in the highest possible test accuracy after 10 epochs.

Your code will be 
"""

GENERATE_HYPOTHESIS = """Study the current version of main.py:

{code}

Then, consider the summary of this implementation and the result of running it. In the summary, the hypothesis value refers to the original hypothesis motivating this implementation.

{summary}

First, summarize at a high level what the current implementation does.
Next, come up with a new hypothesis for how you can improve this code in order to achieve the same or better performance in terms of negative log-likelihood in the shortest number of training steps possible.

Structure your response as a JSON in this format. Do not include any extra commentary in your response.

{{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}}
"""


IMPLEMENT_HYPOTHESIS = """Study the current version of main.py:

{code}

Now implement the following hypothesis for improving the code. 

{hypothesis}

Make sure your code changes preserve these aspects of train_gpt.py:
- The script continues to be runnable via simply calling `python main.py`, so any changes to the hyperparameters in the arguments should be set by directly setting the default value for that argument.
- Do not change the format of the logging print statements.

If you violate any of the above constraints, the experiment run will be invalid.

Respond with only the fully-functional updated code, which implements ideas in the hypothesis above, without any extra commentary. This updated code should be comprehensive of the ideas for improving the code discussed above.
"""
