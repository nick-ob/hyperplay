"""File containing the loss logic.
Represented as a class, as well as metrics represented as functions.

Usage example:

    cce = CCE()
    predicted = np.array([[1], [0], [1]])
    true = np.array([[0], [1], [1]])
    cost = cce.cost(predicted, true)

"""
import numpy as np

class CCE:
    """A class representing the loss calculation of a neural network.
    """
    def cost(self, y_pred: np.ndarray, y_act: np.ndarray) -> float:
        """Compute the cost of the network using the categorial cross entropy loss function.

        Args:
            y_pred: The predicted labels.
            y_act: The actual labels.

        Returns:
            float: The cost of the network.
        """
        # clip data to avoid any logs of 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # shape (batches, categories)

        # shape (batches,), so 1 value for each batch
        cost: np.ndarray = np.sum(y_act * np.log(y_pred), axis=1)

        return -np.mean(cost) # take the mean to be independant of the batch size

    def delta(self, y_pred: np.ndarray, y_act: np.ndarray) -> np.ndarray:
        """Compute the initial delta for backpropagation.
        (the derivative of the loss w.r.t. the final output).

        Args:
            y_pred: The predicted labels.
            y_act: The actual lables.

        Returns:
            np.ndarray: The derviative of the loss w.r.t. the final output.
        """
        # this output averaged to stay independant of the amount of batches
        return (y_pred - y_act) / y_pred.shape[0] # shape (batches, categories)

def accuracy(y_pred: np.ndarray, y_act: np.ndarray) -> float:
    """Compute accuracy of the network.

    Args:
         y_pred: The predicted labels.
         y_act: The actual labels.

    Returns:
        float: The accuracy (in percent).
    """
    # get the indices of the maximum values
    predicted = np.argmax(y_pred, axis=1)
    true = np.argmax(y_act, axis=1)

    n_correct = np.sum(predicted == true)

    return round((n_correct / len(true)) * 100, 2)
