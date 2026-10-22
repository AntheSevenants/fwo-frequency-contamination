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

    def __init__(self, contamination_model: "ContaminationModel", group: int):
        """Initialise an ContaminationModel agent

        Args:
            contamination_model (ContaminationModel):
            The ContaminationModel for which the agent is initialised
            group: The group to which the agent belongs.
        """

        # Pass the parameters to the parent class.
        self.model: "ContaminationModel"
        super().__init__(contamination_model)

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

        if type(self.model.params.base_contaminated) == list:
            base_contaminated = self.model.params.base_contaminated[group]
        elif type(self.model.params.base_contaminated) == float:
            base_contaminated = self.model.params.base_contaminated
        else:
            raise TypeError("base_contaminated parameter is wrong type")

        # Populate the agent's parameters based off the model parameters
        self.atts = model.defaults.agent.Attributes(
            model_params=self.model.params,
            group=group,
            base_contaminated=base_contaminated,
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
        group = self.atts.group if self.model.params.group_distinction else None
        hearer_agent = self.model.get_random_agent(self, group=group)

        # If there *is* an interaction, do interact
        self.interact(hearer_agent)

    def interact(self, hearer_agent: "ContaminationAgent"):
        """This function describes the routine that every agent goes through when they interact.

        Args:
            hearer_agent (ContaminationAgent): The other agent with which the current agent interacts.
        """
        # First, check whether we will communicate construction A or B, or nothing at all
        construction = self.atts.get_construction()

        # Now, let's check whether the agent will produce a contaminated form
        # profiles = (contamination, contamination)
        profiles, counts = self.atts.tally.get_construction_counts(construction)
        threshold = counts[0] / counts.sum()
        contamination = (
            profiles[0]
            if self.model.params.nprandom.random() < threshold
            else profiles[1]
        )

        # "iets verkeerdS"
        if (
            construction == model.enums.Construction.A
            and contamination == model.enums.Contamination.NONE
        ):
            is_ambiguous = False
        else:
            ambiguity_prob = self.atts.get_ambiguity_prob(construction)
            is_ambiguous = self.model.params.nprandom.random() < ambiguity_prob

        hearer_agent.receive_construction(construction, contamination, is_ambiguous)

    def receive_construction(
        self, construction: int, contamination: int, is_ambiguous: bool
    ):
        # if there is contamination, it does not mean there is ambiguity,
        # in some contexts the ambiguity can be overcome
        # but if there is ambiguity, we need to know
        if is_ambiguous:
            # P(A | contamination) ∝ P(contamination | A) * P(A)
            # P(B | contamination) ∝ P(contamination | B) * P(B)

            # The construction, whether or not there is contamination
            real_profile = (construction, contamination)
            # The other construction, inverse contamination
            other_construction = np.abs(construction - 1)
            confusion_profile = (
                other_construction,
                np.abs(contamination - 1),
            )

            real_prob = (
                self.atts.tally.get_contamination_prob_for_construction_introspection(
                    *real_profile
                )
                * self.atts.tally.get_construction_prob_introspection(construction)
            )
            confusion_prob = (
                self.atts.tally.get_contamination_prob_for_construction_introspection(
                    *confusion_profile
                )
                * self.atts.tally.get_construction_prob_introspection(
                    other_construction
                )
            )

            if real_prob >= confusion_prob:
                heard_profile = real_profile
            else:
                heard_profile = confusion_profile
        else:
            heard_profile = (construction, contamination)

        self.atts.tally.update(*heard_profile)
        self.track_communication((construction, contamination), heard_profile)

    def track_communication(
        self, intended_profile: Tuple[int, int], heard_profile: Tuple[int, int]
    ):
        intended_state = model.enums.State.get_state(*intended_profile)
        heard_state = model.enums.State.get_state(*heard_profile)

        self.model.tracker.communicative_success.register_communication_outcome(
            self.atts.group, intended_state, heard_state
        )

    @property
    def is_dead(self):
        # Eternal life
        if self.atts.max_age is None:
            return False

        # Emperor says: you live!
        if self.atts.age < self.atts.max_age:
            return False

        return True
