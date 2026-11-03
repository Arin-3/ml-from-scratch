import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.linear_regression import LinearRegression 

np.random.seed(42)
X = np.random.rand(20,10)
true_w = np.array([3.0, 5.0, 4.0, 1.0, 6.0, 9.0, 2.0, 4.5, 6.7, 8.0])
true_b = 10.0
noise = np.random.randn(20)*5
y = np.dot(X,true_w)+true_b + noise

model = LinearRegression(alpha = 0.1, num_steps = 10000, lambda_=0.2)
model.fit(X,y)

print("Learned w:", model.w)  
print("Learned b:", model.b)
print("Final cost:", model.cost_history[-1])

print("Learned w:", np.round(model.w, 2))
print("True w:   ", true_w)
print("Weight error:", np.sum((model.w - true_w) ** 2))