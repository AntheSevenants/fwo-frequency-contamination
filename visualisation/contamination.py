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
