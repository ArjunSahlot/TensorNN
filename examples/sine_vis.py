import sys
from itertools import count
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tensornn import *
from tensornn.nn import *
from tensornn.activation import *
from tensornn.loss import *
from tensornn.layers import *
from tensornn.optimizers import *
from tensornn.debug import debug

import numpy as np
import matplotlib.pyplot as plt

RANGE = (-np.pi*5, np.pi*5)

def generate_data(n: int = 100):
    x = np.random.uniform(*RANGE, (n, 1))
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
        Dense(64, activation=Sigmoid()),
        Dense(1, activation=Tanh()),
    ])
    nn.register(MSE(), Adam(0.001, 0.9))
    return nn

def train_with_visualization(nn: NeuralNetwork, x_train: Tensor, y_train: Tensor, epochs: int | float = 1_000):
    plt.ion()
    x_plot = Tensor(np.linspace(*RANGE, 300).reshape(-1, 1))

    fig, (ax, loss_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.set_title("Neural network learning sin(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot(x_plot, np.sin(x_plot), label="sin(x)")
    pred_line, = ax.plot(x_plot, np.zeros_like(x_plot), label="NN prediction")
    ax.legend(loc="upper right")

    loss_ax.set_title("Loss history")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("MSE")
    loss_line, = loss_ax.plot([], [], color="tab:red", linewidth=1.5)

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(2)

    loss_history = []
    epoch_iter = count(1) if np.isinf(epochs) else range(1, int(epochs) + 1)
    for epoch in epoch_iter:
        y_pred = nn.forward(x_train)
        loss = nn.loss.calculate(y_pred, y_train)
        loss_value = float(np.mean(loss))
        loss_deriv = nn.loss.derivative(y_pred, y_train)
        nn.backward(loss_deriv)
        nn.optimizer.step()

        loss_history.append(loss_value)

        pred_vals = nn.forward(x_plot)
        pred_line.set_ydata(pred_vals)

        loss_epochs = np.arange(1, len(loss_history) + 1)
        loss_line.set_data(loss_epochs, loss_history)
        loss_ax.relim()
        loss_ax.autoscale_view()

        ax.set_title(f"Epoch {epoch} - MSE: {loss_value:.6f}")

        fig.canvas.draw()
        fig.canvas.flush_events()

def test_sine():
    x_train, y_train = generate_data(500)

    nn = create_nn()

    train_with_visualization(nn, x_train, y_train, epochs=float("inf"))


if __name__ == "__main__":
    test_sine()
