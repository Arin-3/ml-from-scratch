import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.linear_regression import LinearRegression 

np.random.seed(42)
X = np.random.rand(200,3)
true_w = np.array([3.0, 5.0, 4.0])
true_b = 10.0
y = np.dot(X,true_w)+true_b

model = LinearRegression(alpha = 0.1, num_steps = 1000)
model.fit(X,y)

print("Learned w:", model.w)  
print("Learned b:", model.b)
print("Final cost:", model.cost_history[-1])