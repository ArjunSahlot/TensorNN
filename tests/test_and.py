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
        [0],
        [0],
        [0],
        [1],
    ])
    return inputs, outputs

def create_nn():
    nn = NeuralNetwork([
        Input(2),
        Dense(2, activation=Tanh()),
        Dense(1)
    ])
    nn.register(MSE(), Adam(0.02))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_and(seed):
    set_seed(seed)

    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=500)

    pred = nn.forward(inputs)

    assert np.allclose(pred, outputs, atol=0.1)

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_drops_with_training(seed):
    set_seed(seed)
    x, y = generate_data()
    nn = create_nn()
    
    loss0 = np.mean(nn.get_loss(x, y))

    nn.train(x, y, epochs=10)
    
    loss1 = np.mean(nn.get_loss(x, y))
    
    assert loss1 < 0.5 * loss0, f"Loss didn't drop as much as expected, dropped {(loss0 - loss1) / loss0:.2%}"