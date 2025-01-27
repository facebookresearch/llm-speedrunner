from typing import Union

import json
import re

from core import validators


with open('workspaces/collatz_test/v_1/results.json', 'r') as f:
    summary = json.loads(f.read())

metric_types = {k: Union[type(v), None] for k, v in summary.get('metrics', {}).items()}
metric_types_str = json.dumps({k: type(v).__name__ for k, v in summary.get('metrics', {}).items()})

with open('workspaces/collatz_test/v_9/llm_history.jsonl', 'r') as f:
    json_datas = [f.readline() for _ in f]


data = json.loads(json_datas[-1])
metric_response = data['response']

x = '\n\n'.join([metric_response]*3)

x = """
In order to
{
  "runtime": 1.96,
  "start_value": 5857346,
  "max_steps": 382
}

In order to
{
  "runtime": 1.96,
  "start_value": 5857346,
  "max_steps": 382
}

In order to
{
  "runtime": 1.96,
  "start_value": 5857346,
  "max_steps": 382
}
"""

print('x is')
print(x + '\n')

metrics = validators.validate_json(x, metric_types)

print('validated metrics are')
print(metrics)

