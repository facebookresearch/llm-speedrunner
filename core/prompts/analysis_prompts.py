# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SUMMARIZE_LOGS_PROMPT = """Task: Produce a succinct summary of the following stdout and stderr logs for a job running on a compute cluster. 
- Your summary should consider whether the logs indicate whether the goal below was achieved or not.
- Keep your summary below 500 words.

# Job goal
{goal}


# stdout logs
{log_out}


# stderr logs
{log_err}

Respond with just your summary text with no extra commentary and no extra formatting. If appropriate, include the most useful stderr logs for debugging in code blocks fenced by triple ticks.
"""


PARSE_METRICS_FROM_LOGS = """Task: Analyze the following output logs and extract metrics following the metrics structure and typing template provided below. 

# Logs
{logs}

# Metric dict template (showing expected type for each key)
{metric_types}

Respond with only the extracted metrics as a JSON dict following the exact structure and type specification in the dict template below. 
If no metrics are successfully extracted, return the empty dict, {{}}. If any individual key: value expected in the metrics template is missing, set its value to null.
"""
