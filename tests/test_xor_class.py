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
    return inputs, outputs

def create_nn():
    nn = NeuralNetwork([
        Input(2),
        Dense(2, activation=Tanh()),
        Dense(2, activation=Softmax())
    ])
    nn.register(CategoricalCrossEntropy(), Adam(0.01, 0))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_xor(seed):
    set_seed(seed)

    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=2_000)

    loss = np.mean(nn.get_loss(inputs, outputs))
    pred = nn.forward(inputs)

    # Check if the predicted class matches the true class
    predicted_classes = np.argmax(pred, axis=1)
    true_classes = np.argmax(outputs, axis=1)
    assert np.array_equal(predicted_classes, true_classes)
