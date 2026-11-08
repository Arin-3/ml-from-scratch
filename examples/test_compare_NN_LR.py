"""COMPARING LOGISTIC REGRESSION AND 2 LAYER NEURAL NETWORK USING MAKE_MOONS DATASET"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logistic_regression import LogisticRegression
from src.neural_network import TwoLayerNeuralNetwork


def compute_accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(y_hat == y) * 100)


def plot_decision_boundary(
    model: LogisticRegression | TwoLayerNeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    title: str,
) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid_X = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_X).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=25)
    ax.set_title(title)
    ax.set_xlabel("$x_1$ (scaled)")
    ax.set_ylabel("$x_2$ (scaled)")
    ax.grid(True, linestyle="--", alpha=0.4)


def main() -> None:
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=69)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Training Logistic Regression ---")
    log_reg = LogisticRegression(alpha=0.01, steps=100000, lambda_=0.01)
    log_reg.fit(X_train_scaled, y_train, verbose=True)
    lr_test_acc = compute_accuracy(log_reg.predict(X_test_scaled), y_test)

    print("\n--- Training 2-Layer Neural Network ---")
    nn_model = TwoLayerNeuralNetwork(
        hidden_units=32,
        alpha=0.01,
        steps=100000,
        lambda_=0.01,
        random_state=69,
    )
    nn_model.fit(X_train_scaled, y_train, verbose=True)
    nn_test_acc = compute_accuracy(nn_model.predict(X_test_scaled), y_test)

    print(f"\nMoons Test Accuracy | Logistic Regression: {lr_test_acc:.2f}%")
    print(f"Moons Test Accuracy | 2-Layer Neural Net:  {nn_test_acc:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(
        log_reg,
        X_test_scaled,
        y_test,
        axes[0],
        f"Logistic Regression (Test Acc: {lr_test_acc:.1f}%)",
    )
    plot_decision_boundary(
        nn_model,
        X_test_scaled,
        y_test,
        axes[1],
        f"2-Layer NN - 32 ReLU Units (Test Acc: {nn_test_acc:.1f}%)",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()