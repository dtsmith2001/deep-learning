import numpy as np

from activation_functions import sigmoid


class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(self, training_data, epochs: int, mini_batch_size: int, eta: float, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch: list, eta: float):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(b.shape) for b in self.weights]
        n = len(mini_batch)
        eta_over_n = eta/n
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - eta_over_n * nw for w, nw in zip(self.weights, nabla_w)]
        self.baises = [b - eta_over_n * nb for b, nb in zip(self.biases, nabla_b)]
