"""File containing the logic of the network, represented as a class.

Usage example:

    network = Network(3, 2, 2)
    x_train = np.array([1, 0, 1])
    y_train = np.array([[1]])
    network.train((x_train, y_train), 0.01, 100, batch_size=12)
"""
import numpy as np
from src.loss import CCE
from src.layer import Layer
from src.activations import ReLu, Softmax

class Network:
    """A class representing the actual network.

    Pieces together the individual parts (layers, activations, loss) to get a running network.

    Attributes:
        __arch: The architecture of the network, meaning the layers and their node amounts.
        __layers: The layers of the network, including activation layers.
        __history: A variable to cache the training history. Needed to save in the save function.
    """
    def __init__(self, *nodes: int) -> None:
        """Initialises instances using node amounts.

        Args:
            nodes: All node counts. Each node count represents the amount of nodes of one layer.
        """
        # make sure inputs are valid
        if len(nodes) < 2:
            raise ValueError("""Network must have at least 2 layers (input + output).
                             Make sure at least 2 arguments are being passed.""")
        if nodes[-1] < 2:
            raise ValueError("""Last layer must have at least 2 nodes,
                             representing the probability of all possilbe outputs.""")
        for node in nodes:
            if type(node) is not int or node < 1:
                raise ValueError("""All layer node amounts must be positive integers.""")

        self.__arch: tuple[int] = nodes
        self.__layers = self.__init_layers(nodes)
        self.__history: dict[str, list] = {}

    def __init_layers(self, nodes: tuple[int]) -> list:
        """Takes a tuple containing layer nodes and creates a list of actual layers.

        Args:
            nodes: Tuple of integers representing the layers and their node counts.

        Returns:
            list: List of layers representing the network.
        """
        layers: list = []

        # iterate every node count except the last
        # (node count represent the in_nodes in the Layer class)
        for i, in_nodes in enumerate(nodes[:-1]):
            # get node count of next layer (the out nodes for the current layer)
            out_nodes = nodes[i + 1]

            layers.append(Layer(in_nodes, out_nodes))
            layers.append(ReLu())

        layers[-1] = Softmax()
        return layers

    def __forward_feed(self, x: np.ndarray) -> np.ndarray:
        """Forward propagate through all of the layers.

        Args:
            x: The input data to feed through the network.

        Returns:
            np.ndarray: The output of the last layer, i.e. the predicted labels.
        """
        # feed values through all of the layers,
        # with each layer getting the output of the previous layer as its input
        for layer in self.__layers:
            x = layer.forward(x)

        return x

    def __backpropagate(self, delta: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through all network layers.

        Args:
            delta: The derivative of the loss w.r.t. the final output.
            learning_rate: The learning rate for gradient descent.
        """
        # backpropagate through layers, passing on delta to previous layers
        for layer in reversed(self.__layers):
            delta = layer.backward(delta, learning_rate)
            # since there is no use for delta, it is simply discarded

    def __shuffle_data(
            self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle x and y data row-wise so that they still align.

        Args:
            x: The input of the training data.
            y: The labels of the training data.

        Returns:
            tuple: The input and output of the shuffled data.
        """
        # randomly shuffle indices
        indices = np.random.permutation(x.shape[0])

        # apply shuffled indices to data
        return (x[indices], y[indices])

    def train(
            self, data: tuple[np.ndarray, np.ndarray],
            learning_rate: float, epochs: int, batch_size: int | None = None,
            snapshot_interval: int = 10,
            on_snapshot: "callable[[int, int, 'Network'], None] | None" = None,
            should_stop: "callable[[], bool] | None" = None
    ) -> None:
        """Train the network on provided training data.

        Emits snapshots for live visualisation.

        Args:
            data: The input and labels for the training data.
            learning_rate: The learning rate used for gradient descent.
            epochs: The amount of training epochs.
            batch_size: The size of the batches used to train. Defaults to None.
            If None, then the full dataset is used.
            snapshot_interval: Interval (in steps) for emitting snapshots.
            on_snapshot: Callback for live visualisation.
            should_stop: Callback used to cancel training early.
        """
        x, y = data

        # validate inputs
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        if type(snapshot_interval) is not int or snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be a positive integer")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of batches")

        # use full data if no batch size is given
        if batch_size is None:
            batch_size = x.shape[0]

        cce = CCE()

        # training loop
        for epoch in range(1, epochs + 1):
            # shuffle input and output
            x_shuffled, y_shuffled = self.__shuffle_data(x, y)

            # go through all batches for the data
            i: int = 0
            for step, j in enumerate(range(batch_size, x.shape[0] + batch_size, batch_size)):
                # get batch & prepare variables for the next
                x_b, y_b = x_shuffled[i:j], y_shuffled[i:j]
                i = j

                # forward feed and backpropagate through the network
                y_pred = self.__forward_feed(x_b)
                self.__backpropagate(cce.delta(y_pred, y_b), learning_rate)

                # emit snapshot for live visualisation
                if on_snapshot is not None and step % snapshot_interval == 0:
                    on_snapshot(epoch, step, self)

                # stop training if requested
                if should_stop is not None and should_stop():
                    return

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels using the network with the provided data.

        Args:
            x: The input data.

        Returns:
            The predicted labels.
        """
        pred = self.__forward_feed(x)
        return pred
