"""The main entrypoint file of the program.
"""
import numpy as np
from src.loading import load_data
from src.gui import GUI

if __name__ == "__main__":
    np.random.seed(1)

    x, y = load_data("circles.csv")

    gui = GUI()
    gui.mainloop()
