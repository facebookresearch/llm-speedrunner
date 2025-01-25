SCIENTIST_SYSTEM_PROMPT = """You are a machine learning scientist, with expertise in large language models and high-performance computing. Use your expertise to assist the user in their machine learning task.
"""


# NanoGPT speedrun prompts
NANOGPT_TASK_PREAMBLE = """Your goal is to improve the training logic in train_gpt.py to achieve the lowest possible training loss on the FineWeb dataset in the shortest amount of time possible.
"""

NANOGPT_TASK_GENERATE_HYPOTHESIS = """Study the current version of train_gpt.py:

{code}

First, summarize at a high level what the current implementation does.
Next, come up with a hypothesis for how you can improve this code in order to achieve the same or better performance in terms of negative log-likelihood in the shortest number of training steps possible.

Structure your response as a JSON in this format. Do not include any extra commentary in your response.

{
	"summary": Summary of the current implementation,
	"hypothesis": Hypothesis for improving the implementation
}
"""


NANOGPT_TASK_IMPLEMENT_HYPOTHESIS = """Study the current version of train_gpt.py:

{code}

Now implement the following hypothesis for improving the code. 

{hypothesis}

Make sure your code changes preserve these aspects of train_gpt.py:
- The script continues to be runnable via simply calling `python train_gpt.py`.
- Do NOT change the value of train_files, val_files, or val_token values in the Hyperparameters config used to set the training args.
- Always keep save_checkpoint set to False in the training args.
- Keep all print0 statements as is. Do not change the arguments used in the curent print0 statements, so that logging format is preserved.

If you violate any of the above constraints, the experiment run will be invalid.

Respond with only the fully-functional updated code, which implements ideas in the hypothesis above, without any extra commentary.
"""