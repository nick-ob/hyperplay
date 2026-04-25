"""File containing the logic of the activation functions, represented as classes.

Usage example:

    sigmoid = Sigmoid()
    input = np.array([1, 0, 1])
    output = sigmoid.forward(input)
"""
import numpy as np

class ReLu:
    """A class representing the relu activation in a neural network.

    In this code implementation, activation functions are logically implemented
    as a seperate layer in the network.
    Thus, activation functions have the same public functions as the Layer class.

    Attributes:
        __x: Caches the input of the activation function (needed for backpropagation).
    """
    def __init__(self) -> None:
        """Initialises instances.
        """
        self.__x: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the relu output and forward it to the next layer.

        Args:
            x: The input from the previous layer.

        Returns:
            np.ndarray: The input the upcoming layer should recieve.
        """
        # store input for gradient computation
        self.__x = x

        return np.maximum(0, x)

    def backward(self, delta: np.ndarray, _) -> np.ndarray:
        """Compute the new delta and give it to the previous Layer.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            _: A placeholder to be able to use this class the same way as the Layer class.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the output of the previous layer.
        """
        # add derivative of the sigmoid to delta
        return delta * (self.__x > 0).astype(float)

class Softmax:
    """A class representing the softmax activation in a neural network.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax and forward it to the next layer.

        Args:
            x: The input from the previous layer.

        Returns:
            np.ndarray: The input for the upcoming Layer.
        """
        # the input is normalised (subtract each value by the maximum value)
        # to avoid numerical overflow in the softmax function
        # this works since the softmax function is shift-invariant
        x = x - np.max(x, axis=1, keepdims=True) # shape (batches, nodes)

        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, delta: np.ndarray, _) -> np.ndarray:
        """Pass on delta to the previous Layer.

        Since our softmax layer is always the last layer in our network, its delta is the
        derivative of the loss w.r.t. the output of the network, which is calculated in the Loss
        class. This is done so that this class can be called in the same way as the Layer class.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            _: A placeholder to be able to use this class the same way as the Layer class.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the output of the previous layer.
        """
        # simply pass on delta recieved from the Loss class
        return delta
