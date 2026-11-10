import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_neural_network import DeepNeuralNetwork

def compute_accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    return np.mean(y_hat==y)*100
# 1. Generate concentric circles dataset
def main():
    X, y = make_circles(n_samples=100000, noise=0.1, factor=0.5, random_state=99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DeepNeuralNetwork(hidden_layers = (32,16,8), alpha = 0.01, steps = 10000, lambda_ = 0.1, random_state = 99)
    model.fit(X_train_scaled, y_train, verbose = True)
    test_pred = model.predict(X_test_scaled)
    train_pred = model.predict(X_train_scaled)
    print(f"Train Accuracy: {compute_accuracy(train_pred, y_train)}")
    print(f"Test Accuracy: {compute_accuracy(test_pred, y_test)}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    #2D Contour Plot
    x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
    y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    _, grid_A_cache = model._forward(grid_points)
    Z_pred = grid_A_cache[-1].reshape(xx.shape)
    ax[0].contourf(xx, yy, Z_pred, levels=1, cmap="RdBu", alpha=0.6)
    ax[0].scatter(
        X_test_scaled[:, 0],
        X_test_scaled[:, 1],
        c=y_test,
        cmap="RdBu",
        edgecolors="k",
        linewidth=1,
    )
    ax[0].set_title("Deep Neural Network Decision Boundary (make_circles)")
    ax[0].set_xlabel("Feature 1 (Scaled)")
    ax[0].set_ylabel("Feature 2 (Scaled)")
    
    #Learning Curve Plot
    ax[1].plot(model.cost_history, color="blue", linewidth=2)
    ax[1].set_title("Training Cost History")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Cost (BCE + L2 Regularization)")
    ax[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()