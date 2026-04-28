"""File containing the logic of the GUI, represented as a class.

Usage example:
    gui = GUI()
    gui.mainloop()
"""
import threading
import queue

import customtkinter as ctk
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        self.__training_thread: threading.Thread | None = None
        self.__stop_event = threading.Event()

        # training settings
        self.__learning_rate: float = 0.01
        self.__epochs: int = 100
        self.__batch_size: int | None = 32
        self.__snapshot_interval: int = 5

        # initialise widget layout and the animation plot
        self.__setup_layout()
        self.__setup_matplotlib()

        # handle window close
        self.protocol("WM_DELETE_WINDOW", self.__on_close)

    def __setup_layout(self) -> None:
        """Create and lay out GUI widgets.
        """
        # create main container
        self.__root_frame = ctk.CTkFrame(self)
        self.__root_frame.pack(fill="both", expand=True)

        # split layout into left/right panels
        self.__left_panel = ctk.CTkFrame(self.__root_frame, width=260)
        self.__left_panel.pack(side="left", fill="y", padx=12, pady=12)

        self.__right_panel = ctk.CTkFrame(self.__root_frame)
        self.__right_panel.pack(side="right", fill="both", expand=True, padx=12, pady=12)

        # right panel controls row
        self.__right_controls = ctk.CTkFrame(self.__right_panel)
        self.__right_controls.pack(side="top", fill="x")

        self.__train_button = ctk.CTkButton(
            self.__right_controls,
            text="Train",
            command=self.__start_training,
        )
        self.__train_button.pack(side="right", padx=(8, 8))

        self.__reset_button = ctk.CTkButton(
            self.__right_controls,
            text="Reset",
            command=self.__reset_network,
        )
        self.__reset_button.pack(side="right")

        # create plot area
        self.__plot_frame = ctk.CTkFrame(self.__right_panel)
        self.__plot_frame.pack(fill="both", expand=True, padx=12, pady=12)

    def __setup_matplotlib(self) -> None:
        """Initialise the matplotlib/seaborn figure and embed it into the GUI.
        """
        # apply seaborn styling for the plot
        sns.set_theme(style="whitegrid")
        self.__cmap = plt.get_cmap("Set2", 2)

        # create matplotlib figure and axis
        self.__fig = Figure(figsize=(9, 6), dpi=100)
        self.__fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.__ax = self.__fig.add_subplot()
        self.__ax.set_axis_off()
        self.__ax.set_position([0, 0, 1, 1])
        self.__ax.set_xlim(self.__grid_xx.min(), self.__grid_xx.max())
        self.__ax.set_ylim(self.__grid_yy.min(), self.__grid_yy.max())

        # create default decision boundary
        z = np.zeros(self.__grid_xx.shape)

        self.__contour = self.__ax.contourf(
            self.__grid_xx,
            self.__grid_yy,
            z,
            cmap=self.__cmap,
            alpha=0.6,
            zorder=1
        )
        # plot training data with labels
        labels = np.argmax(self.__y_train, axis=1)
        self.__scatter = self.__ax.scatter(
            self.__x_train[:, 0],
            self.__x_train[:, 1],
            c=labels,
            cmap=self.__cmap,
            edgecolors="0.5",
            zorder=2
        )
        self.__update_contour(z)

        # embed matplotlib canvas into tkinter
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self.__plot_frame)
        self.__canvas.draw()
        self.__canvas_widget = self.__canvas.get_tk_widget()
        self.__canvas_widget.pack(fill="both", expand=True)

        # start render loop
        self.after(33, self.__render_tick)

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
        # prevent duplicate training threads
        if self.__training_thread is not None and self.__training_thread.is_alive():
            return

        self.__stop_event.clear()
        self.__train_button.configure(state="disabled")

        # start a new thread
        self.__training_thread = threading.Thread(
            target=self.__training_worker,
            args=(self.__stop_event,),
            daemon=True,
        )
        self.__training_thread.start()

    def __reset_network(self) -> None:
        """Placeholder reset handler (to be implemented)."""
        # request training stop
        self.__stop_event.set()

        # wait briefly for training thread to exit
        if self.__training_thread is not None and self.__training_thread.is_alive():
            self.__training_thread.join(timeout=0.5)

        # reinitialise network
        self.__network: Network = Network(2, 50, 50, 2)

        # clear any pending snapshots
        try:
            while True:
                self.__snapshot_queue.get_nowait()
        except queue.Empty:
            pass

        # reset plot to initial state
        z = np.zeros(self.__grid_xx.shape)
        self.__update_contour(z)
        self.__canvas.draw_idle()

        self.__train_button.configure(state="normal")

    def __training_worker(self, stop_event: threading.Event) -> None:
        """Run training in the background and publish visualisation snapshots.

        Args:
            stop_event: Signal to request a graceful stop for training.
        """
        def on_snapshot(epoch: int, step: int, network: Network) -> None:
            # predict on the fixed grid and reshape into mesh form
            pred = network.predict(self.__grid_xy)
            grid = pred[:, 0].reshape(self.__grid_xx.shape)

            snapshot = TrainingSnapshot(epoch, step, grid)
            self.__publish_snapshot(snapshot)

        def should_stop() -> bool:
            return stop_event.is_set()

        self.__network.train(
            (self.__x_train, self.__y_train),
            self.__learning_rate,
            self.__epochs,
            batch_size=self.__batch_size,
            snapshot_interval=self.__snapshot_interval,
            on_snapshot=on_snapshot,
            should_stop=should_stop,
        )

        # re-enable start button when training is done
        self.__train_button.configure(state="normal")

    def __publish_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Publish the latest snapshot to the UI thread.

        This drops any older snapshot if the queue is full.

        Args:
            snapshot: The latest training state for visualization.
        """
        # ensure only the newest snapshot is kept
        try:
            self.__snapshot_queue.put_nowait(snapshot)
        except queue.Full:
            self.__snapshot_queue.get_nowait()
            self.__snapshot_queue.put_nowait(snapshot)

    def __render_tick(self) -> None:
        """Render loop tick scheduled by Tk's event loop.

        This consumes the latest snapshot (if any), updates the plot artists,
        and schedules the next tick.
        """
        # consume latest snapshot if available
        try:
            snapshot = self.__snapshot_queue.get_nowait()
        except queue.Empty:
            snapshot = None

        if snapshot is not None:
            self.__update_contour(snapshot.grid)
            self.__canvas.draw_idle()

        # schedule next tick
        self.after(33, self.__render_tick)

    def __update_contour(self, grid: np.ndarray) -> None:
        """Update the decision boundary contour using the latest grid values.

        Args:
            grid: Grid of predicted class probabilities reshaped to mesh size.
        """
        # remove old contour before drawing a new one
        self.__contour.remove()

        # flip grid values (neede for colors to align)
        grid = (grid < 0.5).astype(int)

        # update countour
        self.__contour = self.__ax.contourf(
            self.__grid_xx,
            self.__grid_yy,
            grid,
            levels=[0.0, 0.5, 1.0],
            cmap=self.__cmap,
            alpha=0.6,
            zorder=1
        )
        self.__scatter.set_zorder(2)

    def __on_close(self) -> None:
        """Handle window close by stopping background work cleanly."""
        # request training stop
        self.__stop_event.set()

        # wait briefly for training thread to exit
        if self.__training_thread is not None and self.__training_thread.is_alive():
            self.__training_thread.join(timeout=0.5)

        self.destroy()
