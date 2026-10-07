from dataclasses import dataclass

import numpy as np
import scipy.special


@dataclass
class ParsingMode:
    def get_threshold(self, counts: np.ndarray):
        raise NotImplementedError("Please overwrite this method")


@dataclass
class ExactParsing(ParsingMode):
    def get_threshold(self, counts: np.ndarray):
        return counts[0] / counts.sum()


@dataclass
class LazyParsing(ParsingMode):
    temperature: int = 0

    def get_threshold(self, counts: np.ndarray):
        probs = scipy.special.softmax(counts / self.temperature)
        return probs[0]
