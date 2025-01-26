def re


def extract_code(text: str, strict=False):
    pattern = r"```(?:\s*\w+)?\n.*?\n```"
    matches = re.search(pattern, input_string, re.DOTALL)

    if matches:
    	return matches[-1]
   	elif not strict:
   		return text
    else:
    	return ''