import matplotlib.axes
import matplotlib.figure

import model.model
import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_contamination_mean(
    data: Union[model.model.ContaminationModel, List[List[float]]],
    attributes: str | List[str] = "contamination_mean",
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean contamination across agents.

    Args:
        data (Union[model.model.ContaminationModel, List[List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed. Defaults to "contamination_mean".
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        attributes,
        title="Mean contamination per construction across agents",
        **kwargs,
    )


def plot_contamination_mean_per_agent(
    data: Union[model.model.ContaminationModel, List[List[float]]],
    attribute: str = "contamination_mean_per_agent",
    agent_index: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean contamination evolution of a single agent

    Args:
        data (Union[model.model.ContaminationModel, List[List[float]]]): Either a model instance or a list of values.
        attribute (str): The column to fetch data from. Defaults to "contamination_mean_per_agent".
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    return visualisation.core.plot_ratio(
        data,
        attribute,
        agent_filter=agent_index,
        title=f"Mean contamination per construction for agent {agent_index}",
        **kwargs,
    )


def plot_contamination_per_ctx_bar(
    data: model.model.ContaminationModel | List[float],
    attributes: str,
    y_max: int,
    step: int = -1,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean contamination for each per construction across agents

    Args:
        data (List[float]): Contamination per construction. Wrapped in a list because you never know
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        step (int): Step to get the data from. On the scale of the datacollector.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_error_bar(
        data,
        attributes,
        ylim=[0, y_max],
        title=f"Average contamination value (per construction, across agents)",
        y_label="Contamination %",
        x_label="Construction ID",
        step=step,
        **kwargs,
    )


def plot_contamination_per_ctx(
    data: model.model.ContaminationModel | List[List[float]],
    attributes: str,
    filter_dimension: int,
    y_max: int,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean contamination for each per construction across agents

    Args:
        data (List[float]): Contamination per construction. Wrapped in a list because you never know
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        step (int): Step to get the data from. On the scale of the datacollector.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        attributes,
        filter_dimension=filter_dimension,
        ylim=[0, y_max],
        title=f"Average contamination value (per construction, across agents)",
        y_label="Contamination %",
        x_label="Construction ID",
        **kwargs,
    )
