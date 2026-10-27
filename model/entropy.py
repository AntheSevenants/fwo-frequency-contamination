import numpy as np
from scipy.stats import entropy
from typing import List, Optional

ENTROPY_BASE = 2


def fast_entropy(scores: np.ndarray) -> float:
    """Compute entropy (base 2) depending on a probability distribution. Skips scipy.

    Args:
        scores (np.ndarray): The probability distribution. Can be any length.

    Returns:
        float: The computed entropy level.
    """

    p = scores / np.sum(scores)
    p = p[p > 0]

    return -np.sum(p * np.log2(p))


def compute_entropy(probability_distribution: np.ndarray) -> float:
    """Compute entropy (base 2) depending on a probability distribution.

    Args:
        probability_distribution (List[float]): The probability distribution. Can be any length.

    Returns:
        float: The computed entropy level.
    """

    return float(entropy(probability_distribution, base=ENTROPY_BASE))


def compute_maximum_entropy(n: int) -> float:
    """Compute the maximum possible entropy for a given distribution length.

    Args:
        n (int): Distribution length

    Returns:
        float: The computed entropy level.
    """

    return float(entropy(np.ones(n) / n, base=ENTROPY_BASE))
