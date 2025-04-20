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
        Dense(2, activation=Tanh()),
        Dense(1)
    ])
    nn.register(MSE(), SGD(0.05, 0))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_xor(seed):
    set_seed(seed)

    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=10_000)

    loss = np.mean(nn.get_loss(inputs, outputs))
    pred = nn.forward(inputs)

    assert np.allclose(pred, outputs, atol=0.1)

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_drops_with_training(seed):
    set_seed(seed)
    x, y = generate_data()
    nn = create_nn()
    
    loss0 = np.mean(nn.get_loss(x, y))

    nn.train(x, y, epochs=1_000)
    
    loss1 = np.mean(nn.get_loss(x, y))
    
    assert loss1 < 0.5 * loss0, f"Loss didn't drop as much as expected, dropped {(loss0 - loss1) / loss0:.2%}"

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_longer_is_better(seed):
    set_seed(seed)
    x, y = generate_data()
    fast = create_nn()
    slow = create_nn()
    
    slow.train(x, y, epochs=5_000)
    fast.train(x, y, epochs=100)
    
    loss_fast = np.mean(fast.get_loss(x, y))
    loss_slow = np.mean(slow.get_loss(x, y))
    
    assert loss_slow < loss_fast
