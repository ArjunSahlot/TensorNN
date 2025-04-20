"""
This isn't meant to be an official test, rather just a quick test to see if the code works. No
dependency on pytest, no complicated setup, no fixtures, no nothing. Just python and tensornn.

It's ignored in the .gitignore file and will never be updated. It's just a building block for quick
and dirty testing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensornn import *
from tensornn.nn import *
from tensornn.activation import *
from tensornn.loss import *
from tensornn.layers import *
from tensornn.optimizers import *
from tensornn.debug import debug

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
    nn.register(MSE(), SGD(0.05, 0))
    return nn

def test():
    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=8_000)

    pred = nn.forward(inputs)

    for i in range(len(inputs)):
        print(f"Input: {inputs[i]}, Predicted: {pred[i]}, Expected: {outputs[i]}")

if __name__ == "__main__":
    test()