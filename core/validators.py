from typing import Optional, Type
import json
import re


def extract_code(text: str, strict=False):
    pattern = r"```(?:\s*\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1]
    elif not strict:
        return text
    else:
        return ''


def validate_json(x: str, type_dict: Optional[dict[str, Type]] = None) -> Optional[str]:
    print(f"Validating this response as JSON:\n{x}", flush=True)
    data = None

    json_str = extract_code(x, strict=False)

    try:
        data = json.loads(json_str)
    except:
        return None

    if type_dict:
        for k,v in type_dict.items():
            if not k in data or not isinstance(data[k], v):
                return None

    return json_str


def validate_code(x: str) -> Optional[str]:
    print(f"Validating this response as code:\n{x}", flush=True)

    return extract_code(x, strict=False)