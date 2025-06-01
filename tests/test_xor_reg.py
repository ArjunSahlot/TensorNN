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

def generate_data():
    inputs = Tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    outputs = Tensor([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
    ])
    outputs = Tensor([
        [0],
        [1],
        [1],
        [0],
    ])
    return inputs, outputs

def create_nn():
    nn = NeuralNetwork([
        Input(2),
        Dense(4, activation=Tanh()),
        Dense(1)
    ])
    nn.register(MSE(), Adam(0.01))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_xor(seed):
    set_seed(seed)

    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=100)

    loss = np.mean(nn.get_loss(inputs, outputs))
    pred = nn.forward(inputs)

    assert np.allclose(pred, outputs, atol=0.1)
