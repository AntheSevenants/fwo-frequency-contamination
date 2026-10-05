def to_dict(cls):
    original_dict = {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
    reversed_dict = {v: k for k, v in original_dict.items()}

    return reversed_dict


class Ambiguity:
    AMBIGUOUS = 1
    NOT_AMBIGUOUS = 2


class Perspective:
    SOURCE = 0
    TARGET = 1


class Contamination:
    NONE = 0
    CONTAMINATED = 1


class State:
    SOURCE_CLEAN = (Perspective.SOURCE, Contamination.NONE)
    SOURCE_DIRTY = (Perspective.SOURCE, Contamination.CONTAMINATED)
    TARGET_CLEAN = (Perspective.TARGET, Contamination.NONE)
    TARGET_DIRTY = (Perspective.TARGET, Contamination.CONTAMINATED)
