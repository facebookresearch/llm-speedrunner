# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Type
import json
import re


def extract_code(text: str, strict=False) -> Optional[str]:
    pattern = r"```(?:\s*\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1]
    elif not strict:
        return text
    else:
        return ''


def extract_last_json_dict(text: str) -> Optional[str]:
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return None
    
    try:
        last_json = matches[-1]
        return last_json
    except json.JSONDecodeError:
        return None 


def validate_json(x: str, type_dict: Optional[dict[str, Type]] = None) -> Optional[str]:
    print(f"Validating this response as JSON:\n{x}", flush=True)
    data = None

    # First parse out just the last json dict str, as r1 likes to return multiple
    json_str = extract_code(x, strict=False)
    json_str = extract_last_json_dict(json_str)

    try:
        data = json.loads(json_str)
    except:
        print(f"validate_json: Failed to load {json_str}")
        return None

    if type_dict:
        for k,v in type_dict.items():
            if not k in data or not isinstance(data[k], v):
                print(f"validate_json: {k} is not in {data}")
                return None

    return json_str


def validate_code(x: str) -> Optional[str]:
    print(f"Validating this response as code:\n{x}", flush=True)

    return extract_code(x, strict=False)