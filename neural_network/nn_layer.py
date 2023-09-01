from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward_propagation(self, output_error: float, learning_rate: float) -> np.ndarray:
        pass


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights: np.ndarray = np.random.rand(input_size, output_size) - 0.5
        self.bias: np.ndarray = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

