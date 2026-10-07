from itertools import product

import model.enums
import numpy as np
from scipy.stats import norm
from typing import Iterable, List, Tuple
from dataclasses import dataclass


@dataclass
class Vocabulary:
    num_words: int
    priors: List[float]
    priors_enabled: bool

    # # Names to use for the words
    # display_names: List[str]

    def __post_init__(self):
        # self.words: List[str] = [str(word_index) for word_index in self.word_indices]

        # TODO: make these work on a per construction basis
        self.ambiguity_probs: List[float] = [0.5] * self.num_words
        self.a_probs: List[float] = [0.5] * self.num_words

    # TODO: if necessary, make ambiguity construction dependent
    def get_ambiguity_prob(self, word_idx: int, construction: int):
        return self.ambiguity_probs[word_idx]

    def get_A_prob(self, word_idx: int):
        return self.a_probs[word_idx]
