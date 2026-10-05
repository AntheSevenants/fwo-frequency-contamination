def to_dict(cls):
    original_dict = {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
    reversed_dict = {v: k for k, v in original_dict.items()}

    return reversed_dict
