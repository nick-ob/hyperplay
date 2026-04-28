"""The main entrypoint file of the program.
"""
import numpy as np
from src.gui import GUI

if __name__ == "__main__":
    np.random.seed(1)

    gui = GUI()
    gui.mainloop()
