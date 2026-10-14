import mesa
import math
import numpy as np

import model.agent
import model.enums
import model.tracker
import model.reporters
import model.reporters.model
import model.reporters.agent
import model.defaults.model

from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from model.agent import ContaminationAgent


class ContaminationModel(mesa.Model):
    """A model of construction Contamination"""

    def __init__(self, params: model.defaults.model.Parameters):
        """Initialise the Contamination Model

        Args:
            params (model.model_defaults.Parameters):
            Parameters object detailing what parameters the simulation should use
        """

        # Load parameters
        self.params = params

        # Load parent class, set random and seed
        super().__init__(rng=self.params.seed)
        self.nprandom = np.random.default_rng(self.params.seed)

        # Calculate the number of innovators and imitators
        num_a = round(self.params.num_agents * self.params.a_group_share)
        num_b = self.params.num_agents - num_a
        agent_types = [model.enums.Group.GROUP_A] * num_a + [
            model.enums.Group.GROUP_B
        ] * num_b

        # Agents
        agents = model.agent.ContaminationAgent.create_agents(
            model=self, n=self.params.num_agents, group=agent_types
        )

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        # Initialise the agent model reporters
        for_all_types = self.params.a_group_share not in [
            0,
            1,
        ]  # is there a group distinction?
        model_reporters_agents = model.reporters.agent.get_model_reporters(
            for_all_types=for_all_types
        )
        # Initialise the model model reporters
        model_reporters_model = model.reporters.model.get_model_reporters()
        self.datacollector = mesa.DataCollector(
            model_reporters={**model_reporters_agents, **model_reporters_model}
        )

        self.datacollector.collect(self)

    def step(self):
        """Routine run at every step in the simulation"""

        # Make all agents interact in a random order
        self.agents.shuffle_do("interact_do")

        if self.time % self.params.datacollector_step_size == 0:
            # Collect information about this specific model step
            self.datacollector.collect(self)

        # Check if agents need to be replaced
        dead_agents = []
        for agent in self.agents.copy():
            if not agent.is_dead:
                continue

            dead_agents.append(agent)

            # Get a random agent that is not the agent being replaced
            parent_agent = self.get_random_agent(agent)
            # Birth new agent
            model.agent.ContaminationAgent.create_agents(model=self, n=1)

        for dead_agent in dead_agents:
            dead_agent.remove()

    def get_random_agent(
        self, speaker_agent: "ContaminationAgent", group: int | None = None
    ) -> "ContaminationAgent":
        """Choose a random other agnet that is not the agent itself. Pass a group to influence the choice based on preference for the own group.

        Args:
            speaker_agent (ContaminationAgent): The speaker agent. Needs to be passed to prevent communication with self.
            group (int | None): The group to which the agent belongs. When not None, the agent will take into account preference for the own group. Defaults to None.

        Returns:
            ContaminationAgent: The future hearer agent
        """

        # We start with equal distribution btwx the groups
        homophily_weights = [1.0, 1.0]
        if self.params.homophily is not None and group is not None:
            homophily_weights[group] = (
                homophily_weights[group]
                + homophily_weights[group] * self.params.homophily[group]
            )

        # First, filter the candidate agents for eligibility
        candidates: List["ContaminationAgent"] = [
            a for a in self.agents if a != speaker_agent and not a.is_dead
        ]

        if not candidates:
            raise Exception("No agents pool to select from")

        # Assign weights based on group membership
        weights = np.array(
            [1 * homophily_weights[agent.atts.group] for agent in candidates]
        )
        weights /= np.sum(weights)

        # Perform weighted selection
        hearer_agent = self.nprandom.choice(len(candidates), p=weights)

        return candidates[hearer_agent]
