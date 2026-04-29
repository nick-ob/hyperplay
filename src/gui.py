"""File containing the logic of the GUI, represented as a class.

Usage example:
    gui = GUI()
    gui.mainloop()
"""
import threading
import queue
import os
import re

import customtkinter as ctk
ctk.set_default_color_theme("src/theme/marsh.json")
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.model import Network
from src.loss import CCE, accuracy
from src.loading import load_data
from src.types import TrainingSnapshot

class GUI(ctk.CTk):
    """Main application window.

    Owns the GUI layout, training thread lifecycle, and all user-facing state.
    Hyperparameter edits are staged on the left and only applied when the
    Apply button is pressed.
    """
    def __init__(self) -> None:
        """Initialises instances of the GUI class.
        """
        super().__init__()
        self.title("HyperPlay")
        self.iconbitmap("src/theme/icon.ico")
        self.__hidden_layers: list[int] = [10, 10]
        self.__network: Network = Network(2, *self.__hidden_layers, 2)

        # data and decision grid
        self.__data_name: str = "circles.csv"
        self.__x_train, self.__y_train = load_data(self.__data_name)
        self.__grid_xx, self.__grid_yy, self.__grid_xy = self.__create_decision_grid(
            self.__x_train
        )

        self.__snapshot_queue: "queue.Queue[TrainingSnapshot]" = queue.Queue(maxsize=1)
        self.__training_thread: threading.Thread | None = None
        self.__stop_event = threading.Event()

        # training settings (applied)
        self.__learning_rate: float = 0.01
        self.__epochs: int = 100
        self.__batch_size: int | None = 32
        self.__snapshot_interval: int = 5

        # training settings (staged)
        self.__pending_learning_rate: float = self.__learning_rate
        self.__pending_epochs: int = self.__epochs
        self.__pending_batch_size: int | None = self.__batch_size

        self.__loss = CCE()

        # initialise layout and plot
        self.__setup_layout()
        self.__setup_matplotlib()

        # handle window close
        self.protocol("WM_DELETE_WINDOW", self.__on_close)

    def __setup_layout(self) -> None:
        """Create and lay out GUI widgets.

        Layout outline:
            - Left: datasets, architecture, hyperparameters, apply button
            - Right: control row, plot, progress + metrics
        """
        # create main container
        self.__root_frame = ctk.CTkFrame(self)
        self.__root_frame.pack(fill="both", expand=True)

        # left/right layout split
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

        # plot area
        self.__plot_frame = ctk.CTkFrame(self.__right_panel)
        self.__plot_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # right panel status area
        self.__status_frame = ctk.CTkFrame(self.__right_panel)
        self.__status_frame.pack(side="bottom", fill="x", padx=12, pady=(0, 12))

        self.__progress_frame = ctk.CTkFrame(self.__status_frame)
        self.__progress_frame.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.__epoch_label = ctk.CTkLabel(self.__progress_frame, text="Epoch")
        self.__epoch_label.pack(anchor="w", padx=8, pady=(4, 2))
        self.__epoch_progress = ctk.CTkProgressBar(self.__progress_frame)
        self.__epoch_progress.set(0)
        self.__epoch_progress.pack(fill="x", padx=8, pady=(0, 10))

        self.__batch_label = ctk.CTkLabel(self.__progress_frame, text="Batch")
        self.__batch_label.pack(anchor="w", padx=8, pady=(0, 2))
        self.__batch_progress = ctk.CTkProgressBar(self.__progress_frame)
        self.__batch_progress.set(0)
        self.__batch_progress.pack(fill="x", padx=8, pady=(0, 4))

        self.__stats_frame = ctk.CTkFrame(self.__status_frame)
        self.__stats_frame.pack(side="right", padx=12, pady=12)

        self.__cost_frame = ctk.CTkFrame(self.__stats_frame)
        self.__cost_frame.pack(side="left", padx=8, pady=8)
        self.__cost_label = ctk.CTkLabel(self.__cost_frame, text="Cost")
        self.__cost_label.pack(padx=8, pady=(6, 2))
        self.__cost_value = ctk.CTkLabel(self.__cost_frame, text="-")
        self.__cost_value.pack(padx=8, pady=(0, 8))

        self.__accuracy_frame = ctk.CTkFrame(self.__stats_frame)
        self.__accuracy_frame.pack(side="right", padx=8, pady=8)
        self.__accuracy_label = ctk.CTkLabel(self.__accuracy_frame, text="Accuracy")
        self.__accuracy_label.pack(padx=8, pady=(6, 2))
        self.__accuracy_value = ctk.CTkLabel(self.__accuracy_frame, text="-")
        self.__accuracy_value.pack(padx=8, pady=(0, 8))

        # left panel sections
        self.__datasets_section = ctk.CTkFrame(self.__left_panel)
        self.__datasets_section.pack(fill="x", pady=(0, 12))
        self.__datasets_label = ctk.CTkLabel(self.__datasets_section, text="Datasets")
        self.__datasets_label.pack(anchor="w", padx=8, pady=(8, 4))

        self.__dataset_var = ctk.StringVar(value=self.__data_name)
        self.__dataset_menu = ctk.CTkOptionMenu(
            self.__datasets_section,
            values=self.__list_datasets(),
            variable=self.__dataset_var,
        )
        self.__dataset_menu.pack(fill="x", padx=8, pady=(0, 8))

        self.__architecture_section = ctk.CTkFrame(self.__left_panel)
        self.__architecture_section.pack(fill="x", pady=(0, 12))
        self.__architecture_label = ctk.CTkLabel(self.__architecture_section, text="Architecture")
        self.__architecture_label.pack(anchor="w", padx=8, pady=(8, 4))

        self.__arch_entry_frame = ctk.CTkFrame(self.__architecture_section)
        self.__arch_entry_frame.pack(fill="x", padx=8, pady=(0, 8))

        self.__arch_entry_scroll = ctk.CTkScrollbar(
            self.__arch_entry_frame,
            orientation="horizontal",
        )
        self.__arch_entry_scroll.pack(side="bottom", fill="x")

        self.__arch_entry = ctk.CTkEntry(
            self.__arch_entry_frame,
        )
        self.__arch_entry.pack(side="top", fill="x")
        self.__arch_entry.configure(xscrollcommand=self.__arch_entry_scroll.set)
        self.__arch_entry_scroll.configure(command=self.__arch_entry.xview)
        self.__arch_entry.insert(0, self.__format_arch_text())

        self.__arch_entry.configure(
            validate="key",
            validatecommand=(self.register(self.__validate_arch_input), "%P"),
        )

        self.__hypers_section = ctk.CTkFrame(self.__left_panel)
        self.__hypers_section.pack(fill="x")
        self.__hypers_label = ctk.CTkLabel(self.__hypers_section, text="Hypers")
        self.__hypers_label.pack(anchor="w", padx=8, pady=(8, 4))

        self.__lr_value = ctk.StringVar(value=f"{self.__pending_learning_rate:.4f}")
        self.__lr_label = ctk.CTkLabel(self.__hypers_section, text="Learning rate")
        self.__lr_label.pack(anchor="w", padx=8, pady=(4, 2))
        self.__lr_slider = ctk.CTkSlider(
            self.__hypers_section,
            from_=0.0001,
            to=1.0,
            number_of_steps=999,
            command=self.__on_lr_change,
        )
        self.__lr_slider.set(self.__pending_learning_rate)
        self.__lr_slider.pack(fill="x", padx=8, pady=(0, 2))
        self.__lr_value_label = ctk.CTkLabel(self.__hypers_section, textvariable=self.__lr_value)
        self.__lr_value_label.pack(anchor="w", padx=8, pady=(0, 8))

        self.__epochs_value = ctk.StringVar(value=str(self.__pending_epochs))
        self.__epochs_label = ctk.CTkLabel(self.__hypers_section, text="Epochs")
        self.__epochs_label.pack(anchor="w", padx=8, pady=(4, 2))
        self.__epochs_slider = ctk.CTkSlider(
            self.__hypers_section,
            from_=1,
            to=500,
            number_of_steps=499,
            command=self.__on_epochs_change,
        )
        self.__epochs_slider.set(self.__pending_epochs)
        self.__epochs_slider.pack(fill="x", padx=8, pady=(0, 2))
        self.__epochs_value_label = ctk.CTkLabel(self.__hypers_section, textvariable=self.__epochs_value)
        self.__epochs_value_label.pack(anchor="w", padx=8, pady=(0, 8))

        self.__batch_value = ctk.StringVar(value=str(self.__pending_batch_size))
        self.__batch_label = ctk.CTkLabel(self.__hypers_section, text="Batch size")
        self.__batch_label.pack(anchor="w", padx=8, pady=(4, 2))
        self.__batch_slider = ctk.CTkSlider(
            self.__hypers_section,
            from_=1,
            to=256,
            number_of_steps=255,
            command=self.__on_batch_change,
        )
        self.__batch_slider.set(self.__pending_batch_size)
        self.__batch_slider.pack(fill="x", padx=8, pady=(0, 2))
        self.__batch_value_label = ctk.CTkLabel(self.__hypers_section, textvariable=self.__batch_value)
        self.__batch_value_label.pack(anchor="w", padx=8, pady=(0, 8))

        self.__apply_button = ctk.CTkButton(
            self.__left_panel,
            text="Apply",
            command=self.__apply_settings,
        )
        self.__apply_button.pack(fill="x", padx=12, pady=(12, 0))

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
        self.__ax.set_position((0, 0, 1, 1))
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

        # embed matplotlib canvas into customtkinter
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

        Disables Train/Apply, clears any previous stop signal, and launches the
        training worker. Training always uses the last applied settings.
        """
        # prevent duplicate training threads
        if self.__training_thread is not None and self.__training_thread.is_alive():
            return

        self.__stop_event.clear()
        self.__train_button.configure(state="disabled")
        self.__apply_button.configure(state="disabled")
        self.__epoch_progress.set(0)
        self.__batch_progress.set(0)

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
        self.__network: Network = Network(2, *self.__hidden_layers, 2)

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

        self.__epoch_progress.set(0)
        self.__batch_progress.set(0)
        self.__cost_value.configure(text="-")
        self.__accuracy_value.configure(text="-")

        self.__train_button.configure(state="normal")
        self.__apply_button.configure(state="normal")
        self.__train_button.configure(text="Train")

    def __training_worker(self, stop_event: threading.Event) -> None:
        """Run training in the background and publish visualisation snapshots.

        Args:
            stop_event: Signal to request a graceful stop for training.
        """
        def on_snapshot(
            epoch: int,
            step: int,
            network: Network,
            y_pred: np.ndarray,
            y_batch: np.ndarray,
            total_steps: int,
        ) -> None:
            # predict on the fixed grid and reshape into mesh form
            pred = network.predict(self.__grid_xy)
            grid = pred[:, 0].reshape(self.__grid_xx.shape)

            cost_value = self.__loss.cost(y_pred, y_batch)
            acc_value = accuracy(y_pred, y_batch)

            snapshot = TrainingSnapshot(
                epoch,
                step,
                self.__epochs,
                total_steps,
                cost_value,
                acc_value,
                grid,
            )
            self.__publish_snapshot(snapshot)

        def should_stop() -> bool:
            return stop_event.is_set()

        x, y = self.__x_train, self.__y_train

        if self.__batch_size is None:
            batch_size = x.shape[0]
        else:
            batch_size = self.__batch_size

        total_steps = int(np.ceil(x.shape[0] / batch_size))

        self.__network.train(
            (x, y),
            self.__learning_rate,
            self.__epochs,
            batch_size=batch_size,
            snapshot_interval=self.__snapshot_interval,
            on_snapshot=lambda epoch, step, net: on_snapshot(
                epoch,
                step,
                net,
                net.predict(x),
                y,
                total_steps,
            ),
            should_stop=should_stop,
        )

        # re-enable start button when training is done
        self.__train_button.configure(state="normal")
        self.__apply_button.configure(state="normal")
        self.__train_button.configure(text="Continue")

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
            self.__update_training_metrics(snapshot)
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

    def __update_training_metrics(self, snapshot: TrainingSnapshot) -> None:
        """Update progress bars & training statistics.

        Args:
            snapshot: The current snapshot of which metrics should be shown.
        """
        epoch_progress = 0.0
        if snapshot.total_epochs > 0:
            epoch_progress = min(snapshot.epoch / snapshot.total_epochs, 1.0)

        batch_progress = 0.0
        if snapshot.total_steps > 0:
            batch_progress = min((snapshot.step + 1) / snapshot.total_steps, 1.0)

        self.__epoch_progress.set(epoch_progress)
        self.__batch_progress.set(batch_progress)
        self.__cost_value.configure(text=f"{snapshot.cost:.4f}")
        self.__accuracy_value.configure(text=f"{snapshot.accuracy:.2f}%")

    def __list_datasets(self) -> list[str]:
        """Get the available .csv datasets in the data directory.

        Returns:
            list[str]: The available filenames.
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if not os.path.isdir(data_dir):
            return []

        files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        files.sort()
        return files

    def __apply_settings(self) -> None:
        """Apply user defined settings for the network.

        This is the only place where staged UI edits become active training
        settings (dataset, architecture, hyperparameters).
        """
        if self.__training_thread is not None and self.__training_thread.is_alive():
            return

        self.__hidden_layers = self.__parse_arch_text(self.__arch_entry.get())

        self.__learning_rate = self.__pending_learning_rate
        self.__epochs = self.__pending_epochs
        self.__batch_size = self.__pending_batch_size

        selected = self.__dataset_var.get()
        if selected:
            self.__data_name = selected
            self.__x_train, self.__y_train = load_data(self.__data_name)
            self.__grid_xx, self.__grid_yy, self.__grid_xy = self.__create_decision_grid(
                self.__x_train
            )

        self.__cost_value.configure(text="-")
        self.__accuracy_value.configure(text="-")

        self.__epoch_progress.set(0)
        self.__batch_progress.set(0)

        self.__network = Network(2, *self.__hidden_layers, 2)
        self.__train_button.configure(text="Train")

        self.__ax.set_xlim(self.__grid_xx.min(), self.__grid_xx.max())
        self.__ax.set_ylim(self.__grid_yy.min(), self.__grid_yy.max())

        labels = np.argmax(self.__y_train, axis=1)
        self.__scatter.remove()
        self.__scatter = self.__ax.scatter(
            self.__x_train[:, 0],
            self.__x_train[:, 1],
            c=labels,
            cmap=self.__cmap,
            edgecolors="0.5",
            zorder=2,
        )

        z = np.zeros(self.__grid_xx.shape)
        self.__update_contour(z)
        self.__canvas.draw_idle()

    def __parse_arch_text(self, text: str) -> list[int]:
        """Clean the input text from the architecture entry.

        Args:
            text: The text of the entry.

        Returns:
            list[int]: The node counts in a list.
        """
        cleaned = text.strip().strip(",")
        if not cleaned:
            return []

        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        nodes: list[int] = []
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                nodes.append(value)

        return nodes

    def __format_arch_text(self) -> str:
        """Format the architecture into a nice readable format.
        """
        return ", ".join(str(nodes) for nodes in self.__hidden_layers)

    def __validate_arch_input(self, value: str) -> bool:
        """Make sure the input is valid.

        The input goes through a set of simple checks it has to pass to be considered
        valid.

        Args:
            value: The input.

        Returns:
            bool: Whether it is valid or not.
        """
        if value == "":
            return True

        if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*(?:\s*,?\s*)?", value):
            return False

        if ",," in value:
            return False

        if value.startswith(","):
            return False

        if value.startswith(" "):
            return False

        return True

    def __on_lr_change(self, value: float) -> None:
        """Update learning rate.

        value: The learning rate to set.
        """
        self.__pending_learning_rate = max(0.0001, float(value))
        self.__lr_value.set(f"{self.__pending_learning_rate:.4f}")

    def __on_epochs_change(self, value: float) -> None:
        """Update amount of epochs.

        value: The epoch amount to set.
        """
        self.__pending_epochs = max(1, int(round(value)))
        self.__epochs_value.set(str(self.__pending_epochs))

    def __on_batch_change(self, value: float) -> None:
        """Update the batch size.

        value: The batch size to set.
        """
        self.__pending_batch_size = max(1, int(round(value)))
        self.__batch_value.set(str(self.__pending_batch_size))

    def __on_close(self) -> None:
        """Handle window close by stopping background work cleanly."""
        # request training stop
        self.__stop_event.set()

        # wait briefly for training thread to exit
        if self.__training_thread is not None and self.__training_thread.is_alive():
            self.__training_thread.join(timeout=0.5)

        self.destroy()
