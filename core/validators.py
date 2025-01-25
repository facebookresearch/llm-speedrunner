from typing import Optional, Type


# Validators
def validate_json(x: str, type_dict: Optional[dict[str, Type]] = None) -> bool:
	data = None

	try:
		json.loads(x)
	except:
		return False

	if type_dict:
		for k,v in type_dict.items():
			if not k in data or not isinstance(data[k], v):
				return False

	return True