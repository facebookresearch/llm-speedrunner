def basic_type_name_to_type(name: str) -> type:
    type_mapping = {"float": float, "int": int, "str": str, "dict": dict}
    return type_mapping[name]
