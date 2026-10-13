from itertools import product

import model.constructions.sample
import model.constructions.vocab
import model.constructions.parsing
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Type, Any, Tuple

# Mapping of parameter names to their enum classes
PARAMETER_ENUM_MAPPING: Dict[str, Type] = {
    # nothing yet
}


@dataclass
class Parameters:
    # ----
    # Model housekeeping
    # ----
    num_agents: int = 10
    seed: Optional[int] = None

    # After how many steps do we collect data?
    datacollector_step_size: int = 1

    # ----
    # Vocabulary
    # ----
    vocabulary: (
        model.constructions.vocab.FlatVocabulary
        | model.constructions.vocab.PreciseVocabulary
        | model.constructions.vocab.SampleVocabulary
    ) = model.constructions.vocab.FlatVocabulary(100, 0.5, 0.5, 0.5)

    priors_enabled: bool = True

    # ----
    # Agent age
    # ----
    agent_age_mean: int = 0
    agent_age_range: int = 200

    # ----
    # Cognition parameters
    # ----
    parsing_mode: (
        model.constructions.parsing.ExactParsing
        | model.constructions.parsing.LazyParsing
    ) = field(default_factory=lambda: model.constructions.parsing.ExactParsing())

    def __post_init__(self):
        # Initialise random number generator
        self.nprandom = np.random.default_rng(self.seed)

        # Indices
        self.word_indices: List[int] = list(range(0, self.vocabulary.num_words))
