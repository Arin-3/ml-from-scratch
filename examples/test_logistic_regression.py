import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logistic_regression import LogisticRegression

def compute_accuracy(y:np.ndarray, y_hat:np.ndarray)->float:
    """ m = y.shape[0]
    correct = np.count_nonzero(y==y_hat)
    return float((correct/m)*100) """
    return float(np.mean(y==y_hat)*100)

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 14, shuffle = True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(alpha = 0.1, steps = 1000, lambda_=0)
    model.fit(X_train_scaled,y_train,verbose=True)
    y_pred = model.predict(X_test_scaled)
    print(compute_accuracy(y_test, y_pred))
    plt.plot(model.cost_history)
    plt.xlabel("Steps")
    plt.ylabel("Cost")
    plt.title("Logistic Regression Learning Curve")
    plt.grid(True)
    plt.show()
if __name__ == "__main__": 
    main()