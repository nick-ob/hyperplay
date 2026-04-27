"""File containing the logic of the GUI, represented as a class.

Usage example:
    gui = GUI()
    gui.mainloop()
"""
import customtkinter as ctk
from src.model import Network

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
