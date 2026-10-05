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

        # Agents
        agents = model.agent.ContaminationAgent.create_agents(
            model=self, n=self.params.num_agents
        )

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        # Initialise the agent model reporters (no innovator/conservator share in this model)
        model_reporters_agents = model.reporters.agent.get_model_reporters(
            for_all_types=False
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

    def get_random_agent(self, speaker_agent: "ContaminationAgent"):
        # Choose a random other agent that is not the agent itself
        while True:
            hearer_agent = self.random.choice(self.agents)
            if speaker_agent != hearer_agent and not hearer_agent.is_dead:
                break

        return hearer_agent
