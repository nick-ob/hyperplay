"""File containing the logic of computing and creating the needed plots.

Usage example:
    network = Network(...)
    input = ...
    output = ...
    fig = create_plot((input, output), network)
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.model import Network

def create_plot(data: tuple[np.ndarray, np.ndarray], network: Network) -> Figure:
    """Create a plot with data and decision boundary.

    Args:
        data: The original input and output data.
        network: The used neural network to classify.

    Returns:
        plt: The plot showing data and decision boundary.
    """
    x, y = data

    # get plotting area
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    # create a grid over plotting are
    x_range: np.ndarray = np.arange(x_min, x_max, step=0.1)
    y_range: np.ndarray = np.arange(y_min, y_max, step=0.1)
    x_coords, y_coords = np.meshgrid(x_range, y_range)

    # convert the grid to input shaped data
    # to do so, the grid coordinates are flattened and then stacked together
    # to create a shape of (batches, 2)
    grid_points: np.ndarray = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # get predictions of the model for each grid point
    grid_point_predictions: np.ndarray = network.predict(grid_points)
    predictions: np.ndarray = np.argmax(grid_point_predictions, axis=1)
    grid_predictions: np.ndarray = predictions.reshape(x_coords.shape[0], x_coords.shape[1])

    # plot data
    sns.set_theme(style="dark", context="notebook")
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=np.argmax(y, axis=1), palette="Set2", ax=ax)
    ax.contourf(x_coords, y_coords, grid_predictions, levels=[-0.5, 0.5, 1.5], alpha=0.5, cmap="Set2")
    ax.legend().remove()
    ax.set_axis_off()
    fig.tight_layout()

    return fig
