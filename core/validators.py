from typing import Optional, Type
import json


def validate_json(x: str, type_dict: Optional[dict[str, Type]] = None) -> bool:
	data = None

	try:
		data = json.loads(x)
	except:
		return False

	if type_dict:
		for k,v in type_dict.items():
			if not k in data or not isinstance(data[k], v):
				return False

	return True


def validate_code(x: str) -> bool:
    pattern = r"```(?:\s*\w+)?\n.*?\n```"
    return bool(re.search(pattern, input_string, re.DOTALL))