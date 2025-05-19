from tensornn import *
from tensornn.nn import *
from tensornn.activation import *
from tensornn.loss import *
from tensornn.layers import *
from tensornn.optimizers import *
from tensornn.debug import debug

import numpy as np
import pytest

debug.disable()

def generate_data(n=100):
    x = np.random.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(x)
    return Tensor(x), Tensor(y)

def create_nn():
    class Tanh(Activation):
        def forward(self, x: Tensor) -> Tensor:
            return np.tanh(x) * 1.2

        def derivative(self, x: Tensor) -> Tensor:
            return 1.2 * (1 - np.tanh(x) ** 2)

    nn = NeuralNetwork([
        Input(1),
        Dense(64, activation=Sigmoid()),
        Dense(64, activation=Sigmoid()),
        Dense(64, activation=Sigmoid()),
        Dense(1, activation=Tanh()),
    ])
    nn.register(MSE(), Adam(0.002))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sine(seed):
    set_seed(seed)

    x_train, y_train = generate_data(256)
    x_test, y_test   = generate_data(128)

    nn = create_nn()
    
    nn.train(x_train, y_train, epochs=100)

    assert np.isclose(np.mean(nn.get_loss(x_test, y_test)), 0.0, atol=0.1)
