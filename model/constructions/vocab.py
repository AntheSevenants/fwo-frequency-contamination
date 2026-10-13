from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Word:
    prior: float
    ambiguity: float
    a_prob: float


class BaseVocabulary:
    def __init__(self):
        self.words: list[Word] = []
        self.num_words = len(self.words)

    def get_ambiguity_prob(self, word_idx: int, construction: int):
        return self.words[word_idx].ambiguity

    def get_A_prob(self, word_idx: int):
        return self.words[word_idx].a_prob

    def set_priors(self):
        _priors = [word.prior for word in self.words]
        self.priors = np.array(_priors, dtype=np.float64)
        self.priors /= self.priors.sum()


class FlatVocabulary(BaseVocabulary):
    def __init__(
        self,
        num_words: int,
        shared_prior: float,
        shared_ambiguity: float,
        shared_a_prob: float,
    ):
        super().__init__()
        self.num_words = num_words

        for i in range(self.num_words):
            self.words.append(Word(shared_prior, shared_ambiguity, shared_a_prob))

        self.set_priors()


class SampleVocabulary(BaseVocabulary):
    def __init__(
        self, priors: List[float], shared_ambiguity: float, shared_a_prob: float
    ):
        super().__init__()

        for prior in priors:
            self.words.append(Word(prior, shared_ambiguity, shared_a_prob))

        self.set_priors()


class PreciseVocabulary(BaseVocabulary):
    def __init__(self, items: List[Word]):
        super().__init__()
        self.num_words = len(items)

        self.words = items

        self.set_priors()
