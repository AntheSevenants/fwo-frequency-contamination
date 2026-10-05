from __future__ import annotations

import mesa
import model.enums
import model.defaults.agent
import numpy as np
import copy


from scipy.special import logsumexp
from typing import Dict, List, Self, TYPE_CHECKING, Tuple

# from model.success import CommunicationContext

if TYPE_CHECKING:
    from model.model import ContaminationModel


class ContaminationAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(
        self,
        Contamination_model: "ContaminationModel",
    ):
        """Initialise an ContaminationModel agent

        Args:
            Contamination_model (ContaminationModel):
            The ContaminationModel for which the agent is initialised
        """

        # Pass the parameters to the parent class.
        self.model: "ContaminationModel"
        super().__init__(Contamination_model)

        # Compute agent's age
        if self.model.params.agent_age_mean != 0:
            lower_limit = (
                self.model.params.agent_age_mean - self.model.params.agent_age_range
            )
            upper_limit = (
                self.model.params.agent_age_mean + self.model.params.agent_age_range
            )

            age = self.model.nprandom.integers(lower_limit, upper_limit, size=1)[0]
        else:
            age = None

        # Populate the agent's parameters based off the model parameters
        self.atts = model.defaults.agent.Attributes(
            model_params=Contamination_model.params,
            vocabulary=Contamination_model.params.vocabulary,
            max_age=age,
        )

    def interact_do(self):
        """Each timestep, each agent has the opportunity to interact with another random agent.
        This is an outer wrapper script that decides whether the interaction occurs,
        and/or whether there needs to be decay instead.
        """

        # Agent grows one step older
        self.atts.age += 1

        # Choose a random other agent that is not the agent itself
        hearer_agent = self.model.get_random_agent(self)

        # If there *is* an interaction, do interact
        self.interact(hearer_agent)

    def interact(self, hearer_agent: Self):
        """This function describes the routine that every agent goes through when they interact.

        Args:
            hearer_agent (Self): The other agent with which the current agent interacts.
        """
        pass

    def track_communication(self):
        pass

    @property
    def is_dead(self):
        # Eternal life
        if self.atts.max_age is None:
            return False

        # Emperor says: you live!
        if self.atts.age < self.atts.max_age:
            return False

        return True
