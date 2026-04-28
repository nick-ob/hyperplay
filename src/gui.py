"""File containing the logic of the GUI, represented as a class.

Usage example:
    gui = GUI()
    gui.mainloop()
"""
from __future__ import annotations

from typing import Optional
import threading
import queue

import customtkinter as ctk
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.model import Network
from src.loading import load_data
from src.types import TrainingSnapshot

class GUI(ctk.CTk):
    """A class representing the GUI.

    Attributes:
        __network: The neural network used for training, etc.
    """
    def __init__(self) -> None:
        """Initialises instances of the GUI class.
        """
        super().__init__()
        self.title("HyperPlay")
        self.__network: Network = Network(2, 50, 50, 2)

        # load default data and create decision grid
        self.__data_name: str = "circles.csv"
        self.__x_train, self.__y_train = load_data(self.__data_name)
        self.__grid_xx, self.__grid_yy, self.__grid_xy = self.__create_decision_grid(
            self.__x_train
        )

        self.__snapshot_queue: "queue.Queue[TrainingSnapshot]" = queue.Queue(maxsize=1)
        self.__training_thread: Optional[threading.Thread] = None
        self.__stop_event = threading.Event()

        # initialise widget layout and the animation plot
        self.__setup_layout()
        self.__setup_matplotlib()

    def __setup_layout(self) -> None:
        """Create and lay out GUI widgets.
        """
        # create main container
        self.__root_frame = ctk.CTkFrame(self)
        self.__root_frame.pack(fill="both", expand=True)

        # create start button
        self.__start_button = ctk.CTkButton(
            self.__root_frame,
            text="Start",
            command=self.__start_training,
        )
        self.__start_button.pack(side="top")

        # create plot area
        self.__plot_frame = ctk.CTkFrame(self.__root_frame)
        self.__plot_frame.pack(fill="both", expand=True, padx=12, pady=12)

    def __setup_matplotlib(self) -> None:
        """Initialise the matplotlib/seaborn figure and embed it into the GUI.
        """
        # apply seaborn styling for the plot
        sns.set_theme(style="whitegrid")

        # create matplotlib figure and axis
        self.__fig = Figure(figsize=(9, 6), dpi=100)
        self.__fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.__ax = self.__fig.add_subplot()
        self.__ax.set_axis_off()
        self.__ax.set_position([0, 0, 1, 1])

        # create default decision boundary
        z = np.zeros(self.__grid_xx.shape)
        self.__contour = self.__ax.contourf(
            self.__grid_xx,
            self.__grid_yy,
            z,
            cmap="Set2",
            alpha=0.6,
        )

        # plot training data with labels
        labels = np.argmax(self.__y_train, axis=1)
        self.__scatter = self.__ax.scatter(
            self.__x_train[:, 0],
            self.__x_train[:, 1],
            c=labels,
            cmap="Set2",
            edgecolors="0.5",
        )

        # embed matplotlib canvas into tkinter
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self.__plot_frame)
        self.__canvas.draw()
        self.__canvas_widget = self.__canvas.get_tk_widget()
        self.__canvas_widget.pack(fill="both", expand=True)

    def __create_decision_grid(
        self,
        x: np.ndarray,
        padding: float = 0.5,
        resolution: int = 150,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a fixed mesh grid for decision boundary visualisation.

        Args:
            x: Input features used to derive the plot bounds.
            padding: Extra space around the data range.
            resolution: Number of points per axis in the grid.

        Returns:
            tuple: Meshgrid arrays (xx, yy) and flattened grid points (grid_xy).
        """
        # get area to create grid over
        x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
        y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding

        # create grid and convert into network friendly input shape
        x_range = np.linspace(x_min, x_max, resolution)
        y_range = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
        return xx, yy, grid_xy

    def __start_training(self) -> None:
        """Start the background training thread and the render loop.

        This disables the start button, clears any previous stop signal,
        and launches the training worker.
        """
        pass

    def __training_worker(self, stop_event: threading.Event) -> None:
        """Run training in the background and publish visualisation snapshots.

        Args:
            stop_event: Signal to request a graceful stop for training.
        """
        pass

    def __publish_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Publish the latest snapshot to the UI thread.

        This drops any older snapshot if the queue is full.

        Args:
            snapshot: The latest training state for visualization.
        """
        pass

    def __render_tick(self) -> None:
        """Render loop tick scheduled by Tk's event loop.

        This consumes the latest snapshot (if any), updates the plot artists,
        and schedules the next tick.
        """
        pass

    def __update_contour(self, grid: np.ndarray) -> None:
        """Update the decision boundary contour using the latest grid values.

        Args:
            grid: Grid of predicted class probabilities reshaped to mesh size.
        """
        pass

    def __on_close(self) -> None:
        """Handle window close by stopping background work cleanly."""
        pass
