import mesa
import numpy as np
import model.success

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ContaminationModel


def get_nested_attr(obj: object, attr_path: str) -> Any:
    """
    Get a nested attribute from an object using dot notation.

    Args:
        obj: The object to get attributes from.
        attr_path: The attribute path in dot notation (e.g., "activation.level").

    Returns:
        The value of the nested attribute.
    """
    attributes = attr_path.split(".")
    current = obj
    try:
        for attr in attributes:
            current = getattr(current, attr)
        return current
    except AttributeError as e:
        raise AttributeError(f"Attribute '{attr_path}' not found") from e


class Tracker:
    """The tracker is the facilitates keeping track of the current model state."""

    def __init__(self, Contamination_model: "ContaminationModel"):
        """Initialise a Tracker by supplying a MESA model.

        Args:
            Contamination_model (ContaminationModel): A MESA model
        """

        # This will break pickling the model forever. Good :-)
        self.model = Contamination_model
        self.communicative_success = model.success.CommunicativeSuccess(
            Contamination_model
        )

        self.reset()

    # Per step!
    def reset(self):
        self.communicative_success.reset()

        self.chosen_constructions = np.zeros(
            self.model.params.vocabulary.num_constructions
        )
        self.confusion_matrix = np.zeros(
            (
                self.model.params.vocabulary.num_constructions,
                self.model.params.vocabulary.num_constructions,
            )
        )

        self.decision_entropy = []

    def register_construction_chosen(self, construction_idx: int):
        """Register in the tracker which construction was chosen by an agent.

        Args:
            construction_idx (int): The index of the chosen construction
        """

        self.chosen_constructions[construction_idx] += 1

    def register_win_index(self, win_index: int, true_index: int):
        self.confusion_matrix[true_index][win_index] += 1

    def register_decision_entropy(self, decision_entropy: float):
        self.decision_entropy.append(decision_entropy)

    @property
    def __any_outcomes__(self):
        return self.communicative_success.outcomes[
            model.success.CommunicationContext.ANY
        ]

    def get_global(self, property_name):
        return getattr(self, property_name)

    def get_global_property_percentages(
        self, property_name: str, aggregate: bool = False
    ):
        # Get the value of this property
        counts = getattr(self, property_name)

        if aggregate and len(counts.shape) > 1:
            counts = np.sum(counts, axis=0)

        # Turn into percentages
        with np.errstate(divide="ignore", invalid="ignore"):
            if aggregate:
                shares_per_ctx = np.true_divide(counts, np.sum(counts))
                shares_per_ctx = np.nan_to_num(shares_per_ctx)
            else:
                shares_per_ctx = np.true_divide(
                    counts, np.sum(counts, axis=1, keepdims=True)
                )

        # For now I'm only returning the shares as-is.
        # The legend is added afterwards anyway
        return shares_per_ctx

    def get_global_property_mean(self, property_name: str):
        value = getattr(self, property_name)

        if len(value) == 0:
            return 0

        return np.mean(value)

    def get_global_property_median(self, property_name: str):
        value = getattr(self, property_name)

        if len(value) == 0:
            return 0

        return np.median(value)

    def get_property_per_agent(self, property_name: str, index: int | None = None):
        """Retrieve a list of property values of each agent. If a property is multi-dimensional, you can ask to retrieve the value of one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            np.Array: A numpy array containing the requested values.
        """

        agent_property_dist = []

        # Get the property for each agent
        for agent in self.model.agents:
            property_value = get_nested_attr(agent.atts, property_name)
            if index is not None:
                property_value = property_value[index]
            agent_property_dist.append(property_value)

        # Turn into numpy array
        return np.array(agent_property_dist)

    def get_property_mean_across_agents(
        self,
        property_name: str,
        index: int | None = None,
    ):
        """Retrieve the mean of a requested property value across agents. If a property is multi-dimensional, you can ask to take the mean of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the mean of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index)

        return agent_property_dist.mean(axis=0)

    def get_property_median_across_agents(
        self,
        property_name: str,
        index: int | None = None,
    ):
        """Retrieve the median of a requested property value across agents. If a property is multi-dimensional, you can ask to take the median of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the median of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index)

        return np.median(agent_property_dist, axis=0)

    def get_percentage_macro_mean(self, property_name: str):
        percentage_share = self.get_global_property_percentages(property_name)

        mean = np.nanmean(percentage_share, axis=0)
        return np.nan_to_num(mean)

    def get_percentage_macro_median(self, property_name: str):
        percentage_share = self.get_global_property_percentages(property_name)

        median = np.nanmedian(percentage_share, axis=0)
        return np.nan_to_num(median)
