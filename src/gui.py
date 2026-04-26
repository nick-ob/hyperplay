"""File containing the logic of the GUI, represented as a class.

Usage example:
    gui = GUI()
    gui.mainloop()
"""
import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from src.model import Network
from src.drawer import create_plot

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

    def render_plot(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """Renders a given plot in the GUI window.

        Args:
            data: The input and output of the original data
        """
        fig: Figure = create_plot(data, self.__network)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack()
