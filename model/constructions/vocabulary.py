from itertools import product

import model.enums
import numpy as np
from scipy.stats import norm
from typing import Iterable, List, Tuple
from dataclasses import dataclass


@dataclass
class Vocabulary:
    num_constructions: int
    priors: List[float]
    priors_enabled: bool

    # # Names to use for the words
    # display_names: List[str]

    def __post_init__(self):
        self.word_indices: List[int] = list(range(0, self.num_constructions))
        self.words: List[str] = [str(word_index) for word_index in self.word_indices]

        self.frequency: np.ndarray = np.zeros(
            (self.num_constructions, len(model.enums.to_dict(model.enums.State)))
        )

    def get_word(self, word_idx: int) -> str:
        return self.words[word_idx]
