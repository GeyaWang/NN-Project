from . import nn_layer
from typing import Callable
import numpy as np
import pickle
import os


class Network:
    def __init__(self, layers: list[nn_layer.Layer, ...]):
        self.layers = layers
        self.loss_func = None
        self.loss_func_prime = None

    @classmethod
    def load(cls, file_path: str | os.PathLike):
        if not file_path.endswith('.pikl'):
            raise InvalidFileType
        with open(file_path, 'rb') as f:
            return cls(pickle.load(f))

    def save(self, file_path: str | os.PathLike) -> None:
        if not file_path.endswith('.pikl'):
            raise InvalidFileType
        with open(file_path, 'wb') as f:
            pickle.dump(self.layers, f)

    def set_loss_func(self, loss_func: Callable, loss_func_prime: Callable):
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs, learning_rate) -> None:
        if self.loss_func is None or self.loss_func_prime is None:
            raise UndefinedLossFunction

        samples = len(x_train)

        for i in range(epochs):
            display_error = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                display_error = self.loss_func(y_train[j], output)

                # backward propagation
                error = self.loss_func_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            display_error /= samples
            print(f'epoch {i+ 1}/{epochs}   error={display_error}')

    def predict(self, input_data: np.ndarray) -> list[float, ...]:
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result


class UndefinedLossFunction(Exception):
    def __init__(self):
        super().__init__("The loss function is not defined")


class InvalidFileType(Exception):
    def __init__(self):
        super().__init__("File must be pickle file type")
