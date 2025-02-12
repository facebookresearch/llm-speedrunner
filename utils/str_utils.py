import json


def basic_type_name_to_type(name: str) -> type:
    type_mapping = {"float": float, "int": int, "str": str, "dict": dict}
    return type_mapping[name]


def get_serializable_dict_subset(data: dict):
    safe_subset = {}
    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            continue
        else:
            safe_subset[key] = value
    return safe_subset
