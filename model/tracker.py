import mesa
import numpy as np
import model.success
import model.enums

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ContaminationModel
    from model.agent import ContaminationAgent


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

    def __init__(self, contamination_model: "ContaminationModel"):
        """Initialise a Tracker by supplying a MESA model.

        Args:
            Contamination_model (ContaminationModel): A MESA model
        """

        # This will break pickling the model forever. Good :-)
        self.model = contamination_model
        self.communicative_success = model.success.CommunicativeSuccess(
            contamination_model
        )

        self.reset()

    # Per step!
    def reset(self):
        self.communicative_success.reset()
        self.decision_entropy = []

    def register_decision_entropy(self, decision_entropy: float):
        self.decision_entropy.append(decision_entropy)

    @property
    def __confusion_matrix_a__(self):
        return self.communicative_success.get_confusion_matrix_by_group(
            model.enums.Group.GROUP_A
        )

    @property
    def __confusion_matrix_b__(self):
        return self.communicative_success.get_confusion_matrix_by_group(
            model.enums.Group.GROUP_B
        )

    @property
    def __confusion_matrix__(self):
        return self.communicative_success.get_confusion_matrix_normalised(
            [self.model.params.a_group_share, 1 - self.model.params.a_group_share]
        )

    @property
    def __ambiguity_avoidance_rate__(self):
        intended_state = model.enums.State.get_state(
            model.enums.Construction.A, model.enums.Contamination.CONTAMINATED
        )
        confusion_state = model.enums.State.get_state(
            model.enums.Construction.B, model.enums.Contamination.NONE
        )

        a_cont_count = self.__confusion_matrix__[intended_state, intended_state]
        b_none_count = self.__confusion_matrix__[intended_state, confusion_state]

        return float(np.divide(b_none_count, a_cont_count + b_none_count))

    @property
    def __constructional_contamination_rate__(self):
        intended_state = model.enums.State.get_state(
            model.enums.Construction.B, model.enums.Contamination.NONE
        )
        confusion_state = model.enums.State.get_state(
            model.enums.Construction.A, model.enums.Contamination.CONTAMINATED
        )

        b_none_count = self.__confusion_matrix__[intended_state, intended_state]
        a_cont_count = self.__confusion_matrix__[intended_state, confusion_state]

        return float(np.divide(a_cont_count, b_none_count + a_cont_count))

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

    def get_property_per_agent(
        self, property_name: str, group: int | None = None, index: int | None = None
    ):
        """Retrieve a list of property values of each agent. If a property is multi-dimensional, you can ask to retrieve the value of one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            group (int): What group to get this property for. Defaults to None (= all).
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            np.Array: A numpy array containing the requested values.
        """

        agent_property_dist = []

        # Get the property for each agent
        for agent in self.model.agents:
            # For the type checker
            agent: "ContaminationAgent"

            # Only get the correct type
            if group is not None and agent.atts.group != group:
                continue

            property_value = get_nested_attr(agent.atts, property_name)
            if index is not None:
                property_value = property_value[index]
            agent_property_dist.append(property_value)

        # Turn into numpy array
        return np.array(agent_property_dist)

    def get_property_mean_across_agents(
        self,
        property_name: str,
        group: int | None = None,
        index: int | None = None,
    ):
        """Retrieve the mean of a requested property value across agents. If a property is multi-dimensional, you can ask to take the mean of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            group (int): What group to get this property for. Defaults to None (= all).
            index (int): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the mean of the value
        """

        agent_property_dist = self.get_property_per_agent(
            property_name, index=index, group=group
        )

        # Check distribution before running mean
        if agent_property_dist.size == 0:
            raise ValueError(
                "The group for which the property mean was computed is empty."
            )

        return agent_property_dist.mean(axis=0)

    def get_property_median_across_agents(
        self,
        property_name: str,
        group: int | None = None,
        index: int | None = None,
    ):
        """Retrieve the median of a requested property value across agents. If a property is multi-dimensional, you can ask to take the median of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            group (int): What group to get this property for. Defaults to None (= all).
            index (int): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the median of the value
        """

        agent_property_dist = self.get_property_per_agent(
            property_name, index=index, group=group
        )

        # Check distribution before running mean
        if agent_property_dist.size == 0:
            raise ValueError(
                "The group for which the property mean was computed is empty."
            )

        return np.median(agent_property_dist, axis=0)

    def get_percentage_macro_mean(self, property_name: str):
        percentage_share = self.get_global_property_percentages(property_name)

        mean = np.nanmean(percentage_share, axis=0)
        return np.nan_to_num(mean)

    def get_percentage_macro_median(self, property_name: str):
        percentage_share = self.get_global_property_percentages(property_name)

        median = np.nanmedian(percentage_share, axis=0)
        return np.nan_to_num(median)
