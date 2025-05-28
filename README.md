<h1 align="center">
  <img alt="TensorNN logo" src="https://github.com/ArjunSahlot/TensorNN/blob/main/assets/TensorNN-logos_transparent.png?raw=true" width="500px"/><br/>
</h1>

[![GitHub license](https://img.shields.io/github/license/ArjunSahlot/TensorNN)](https://github.com/ArjunSahlot/TensorNN/blob/main/LICENSE)
[![Read the Docs](https://readthedocs.org/projects/tensornn/badge/?version=latest)](https://tensornn.readthedocs.io/en/latest/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/ArjunSahlot/TensorNN)](https://github.com/ArjunSahlot/TensorNN/releases/latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/TensorNN)](https://pypi.org/project/TensorNN/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/TensorNN)](https://pypi.org/project/TensorNN/)

**TensorNN** is a Python machine learning library built from scratch. It provides fundamental building blocks for creating and training neural networks.

## ⚠️ Status: Under Heavy Development

**Note:** This library is currently under heavy development and is considered unusable for production purposes.

---

## 📋 Table of Contents

* [Installation](#installation)
* [Features](#features)
* [Quick Start](#quick-start)
* [Documentation](#documentation)
* [Running Tests](#running-tests)
* [License](#license)
* [Author](#author)

---

## ⚙️ Installation

You can install TensorNN via pip (once it's published):

```bash
pip install tensornn
````

### Dependencies

TensorNN requires the following Python packages:

  * numpy
  * tqdm

For documentation, additional packages are required:

  * sphinx
  * sphinx\_rtd\_theme

-----

## ✨ Features

TensorNN provides a range of modules to facilitate the creation and training of neural networks:

  * **Tensor Operations (`tensornn.tensor`):** Custom `Tensor` class based on NumPy for multi-dimensional array operations.
  * **Neural Network Core (`tensornn.nn`):** The `NeuralNetwork` class to define and manage network architecture.
  * **Layers (`tensornn.layers`):**
      * `Input`: Specifies the input shape to the network.
      * `Dense`: Fully connected layer.
  * **Activation Functions (`tensornn.activation`):** A variety of activation functions to introduce non-linearity, including:
      * `ReLU`
      * `Sigmoid`
      * `Softmax`
      * `Tanh`
      * `LeakyReLU`
      * `ELU`
      * `Swish`
      * `LecunTanh`
      * `NoActivation` (Linear)
  * **Loss Functions (`tensornn.loss`):** Various functions to evaluate model performance:
      * `MSE` (Mean Squared Error)
      * `CategoricalCrossEntropy`
      * `BinaryCrossEntropy`
      * `RMSE` (Root Mean Squared Error)
      * `MAE` (Mean Absolute Error)
      * `MSLE` (Mean Squared Logarithmic Error)
      * `Poisson`
      * `SquaredHinge`
      * `RSS` (Residual Sum of Squares)
  * **Optimizers (`tensornn.optimizers`):** Algorithms to update network weights:
      * `SGD` (Stochastic Gradient Descent)
      * `Adam`
      * `RMSProp`
  * **Utilities (`tensornn.utils`):** Helper functions like `one_hot` encoding, data `normalize`, `set_seed` for reproducibility, `flatten`, and `atleast_2d`.
  * **Debugging (`tensornn.debug`):** Tools to help inspect and debug your network's behavior, including progress bars and logging.
  * **Custom Errors (`tensornn.errors`):** Specific error types for easier debugging (e.g., `NotRegisteredError`, `InitializationError`, `InputDimError`).

-----

## 🚀 Quick Start

Here's a simple example of how to define and train a neural network to solve the XOR problem:

```python
from tensornn import Tensor, NeuralNetwork, Input, Dense, Tanh, Softmax, MSE, CategoricalCrossEntropy, SGD, Adam, set_seed

# For reproducibility
set_seed(0)

# 1. Define the data
inputs = Tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
# For regression (outputting a single value)
# outputs_reg = Tensor([[0], [1], [1], [0]])

# For classification (outputting probabilities for each class)
outputs_class = Tensor([
    [1, 0],  # Class 0
    [0, 1],  # Class 1
    [0, 1],  # Class 1
    [1, 0],  # Class 0
])

# 2. Create the Neural Network
# Example for classification using Softmax and CategoricalCrossEntropy
nn_classifier = NeuralNetwork([
    Input(2),                             # Input layer with 2 features
    Dense(2, activation=Tanh()),        # Hidden layer with 2 neurons and Tanh activation
    Dense(2, activation=Softmax())      # Output layer with 2 neurons (for 2 classes) and Softmax activation
])

# 3. Register the network with a loss function and an optimizer
nn_classifier.register(CategoricalCrossEntropy(), Adam(learning_rate=0.01))

# 4. Train the network
print("Training XOR classifier...")
nn_classifier.train(inputs, outputs_class, epochs=2000, batch_size=4)

# 5. Make predictions
predictions = nn_classifier.forward(inputs)
predicted_classes = predictions.argmax(axis=1)

print("\nClassifier Predictions:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Predicted probabilities: {predictions[i]}, Predicted class: {predicted_classes[i]}, Expected class: {outputs_class[i].argmax()}")
```


You can find more examples in the `examples/` directory, such as `sine_vis.py` which demonstrates training a network to learn the sine function with real-time visualization.

-----

## 📚 Documentation

The official documentation for TensorNN is hosted on ReadTheDocs:

[**https://tensornn.readthedocs.io/en/latest/**](https://www.google.com/url?sa=E&source=gmail&q=https://tensornn.readthedocs.io/en/latest/)

The documentation is built using Sphinx from the files in the `docs/` directory. Key configuration files include `.readthedocs.yaml` and `docs/conf.py`.

-----

## 🧪 Running Tests

TensorNN uses `pytest` for testing. Tests are located in the `tests/` directory. Examples include:

  * `test_and.py`: Tests the AND gate logic.
  * `test_xor_class.py`: Tests XOR classification.
  * `test_xor_reg.py`: Tests XOR regression.
  * `test_linear_regression.py`: Tests simple linear regression.
  * `test_sine.py`: Tests learning the sine function.

To run the tests, you would typically navigate to the root directory of the project and run:

```bash
pytest
```

*(Note: Ensure you have pytest installed: `pip install pytest`)*

-----

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for more details.

-----

## 👨‍💻 Author

  * **Arjun Sahlot** ([iarjun.sahlot@gmail.com](mailto:iarjun.sahlot@gmail.com))
  * GitHub: [ArjunSahlot](https://github.com/ArjunSahlot)
