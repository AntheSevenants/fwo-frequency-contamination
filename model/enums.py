def to_dict(cls):
    original_dict = {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
    reversed_dict = {v: k for k, v in original_dict.items()}

    return reversed_dict


class Ambiguity:
    AMBIGUOUS = 1
    NOT_AMBIGUOUS = 2


class Construction:
    A = 0
    B = 1


class Perspective:
    SOURCE = 0
    TARGET = 1


class Contamination:
    NONE = 0
    CONTAMINATED = 1


class State:
    A_NORMAL = (Construction.A, Contamination.NONE)
    A_CONTAMINATED = (Construction.A, Contamination.CONTAMINATED)
    B_NORMAL = (Construction.B, Contamination.NONE)
    B_CONTAMINATED = (Construction.B, Contamination.CONTAMINATED)

    # Create a lookup table
    _LOOKUP = {
        A_NORMAL: 0,
        A_CONTAMINATED: 1,
        B_NORMAL: 2,
        B_CONTAMINATED: 3,
    }

    _COUNT = 4

    @classmethod
    def get_state(cls, construction, contamination):
        # Using a tuple as a key allows for a single lookup
        return cls._LOOKUP.get((construction, contamination))
