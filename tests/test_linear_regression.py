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

def generate_data(n=100, w=3, b=-4):
    x = np.random.uniform(-1, 1, (n, 1))
    y = w * x + b
    return Tensor(x), Tensor(y)

def create_nn():
    nn = NeuralNetwork([
        Input(1),
        Dense(1)
    ])
    nn.register(MSE(), SGD(0.05, 0))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_regression(seed):
    set_seed(seed)

    weight = 3
    bias = -4

    inputs, outputs = generate_data(10, weight, bias)
    nn = create_nn()

    nn.train(inputs, outputs, epochs=1_000)

    pred = nn.forward(inputs)

    assert np.allclose(pred, outputs, atol=0.1)

    # Check if the weights and biases are close to the expected values
    assert np.allclose(nn.layers[1].weights, weight, atol=0.1)
    assert np.allclose(nn.layers[1].biases, bias, atol=0.1)
