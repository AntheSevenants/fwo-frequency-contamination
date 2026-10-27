import matplotlib.axes
import matplotlib.figure

import model.model
import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_contamination(
    data: Union[model.model.ContaminationModel, List[List[float]]],
    attributes: str | List[str] = "contamination_mean",
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean or median contamination across agents.

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


def plot_ambiguity_avoidance_rate(
    data: Union[model.model.ContaminationModel, List[float]],
    attributes: str | List[str] = "ambiguity_avoidance_rate",
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean or median ambiguity avoidance rate across agents.

    Args:
        data (Union[model.model.ContaminationModel, List[List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed. Defaults to "ambiguity_avoidance_rate_mean".
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=[0, 1],
        y_baseline=0.5,
        title="Mean ambiguity avoidance rate per construction across agents",
        **kwargs,
    )


def plot_contamination_rate(
    data: Union[model.model.ContaminationModel, List[float]],
    attributes: str | List[str] = "constructional_contamination_rate",
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean or median contamination rate across agents.

    Args:
        data (Union[model.model.ContaminationModel, List[List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed. Defaults to "constructional_contamination_rate_mean".
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=[0, 1],
        y_baseline=0.5,
        title="Mean contamination rate per construction across agents",
        **kwargs,
    )
