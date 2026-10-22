import matplotlib.axes
import matplotlib.figure

import model.model
import model.entropy
import visualisation.core

from typing import Optional, List, Union, Tuple, Any


def ylim_from_num_constructions(num_constructions: int):
    """Derive the ylim values from the model values

    Args:
        num_constructions (int): How many constructions appear in the model

    Returns:
        List[float]: The minimum and maximum values for entropy for these model parameters
    """
    ylim = [0, model.entropy.compute_maximum_entropy(num_constructions)]

    return ylim


def plot_ctx_entropy_mean(
    data: Union[model.model.ContaminationModel, List[float]],
    attributes: str | List[str],
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean entropy across agents

    Args:
        data (Union[model.model.ContaminationModel, List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        is_base_rate (bool, optional): Whether the entropy measures are derived from the base rate. Defaults to False.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """
    ylim = ylim_from_num_constructions(2)

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=ylim,
        title=f"Mean entropy across agents",
        **kwargs,
    )
