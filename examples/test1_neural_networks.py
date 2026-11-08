import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.neural_network import TwoLayerNeuralNetwork

def compute_accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    return np.mean(y_hat==y)*100

def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = TwoLayerNeuralNetwork(hidden_units = 16, alpha = 0.01, steps = 10000, lambda_ = 1, random_state = 42)
    model.fit(X_train_scaled, y_train, verbose = True)
    y_hat = model.predict(X_test_scaled)
    
    accuracy = compute_accuracy(y_hat, y_test)
    print(f"accuracy = {accuracy}")
    
    plt.plot(model.cost_history)
    plt.xlabel("Steps")
    plt.ylabel("Cost")
    plt.title("2 Layer Neural Network Learning Curve")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()