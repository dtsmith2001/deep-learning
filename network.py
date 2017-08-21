import numpy as np


class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = None