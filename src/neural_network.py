import numpy as np

class  TwoLayerNeuralNetwork:
    def __init__(self, hidden_units: int = 16, alpha: float = 0.01,steps: int = 1000, lambda_: float = 0.0, random_state: int | None = None) -> None:
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.steps = steps
        self.lambda_ = lambda_
        self.random_state = random_state
        
        self.W1:np.ndarray | None = None
        self.b1:np.ndarray | None = None
        self.W2:np.ndarray | None = None
        self.b2:np.ndarray | None = None
        self.cost_history: list[float] = []
    def _init_params(self, n_features:int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.W1 = rng.standard_normal((n_features, self.hidden_units))*0.01
        self.b1 = np.zeros((1, self.hidden_units))
        self.W2 = rng.standard_normal((self.hidden_units, 1))*0.01
        self.b2 = np.zeros((1,1))
    def _relu(self, Z:np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)  # instead of np.where(Z<0, 0, Z)
    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Z_clipped = np.clip(Z, -500, 500)
        return 1/(1+np.exp(-Z_clipped))
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None): # using "is" instead of "=="
            raise ValueError("Parameters not defined")
        Z1 = X @ self.W1 + self.b1 #(m,n)*(n,h)->(m,h) size  | *FIRST HIDDEN LAYER*
        A1 = self._relu(Z1) 
        Z2 = A1 @ self.W2 + self.b2 #(m,h)*(h,1)->(m,1) size | *SECOND HIDDEN LAYER* or *OUTPUT LAYER*
        A2 = self._sigmoid(Z2) #OUTPUT
        return (Z1, A1, Z2, A2)
    def _compute_cost(self, A2: np.ndarray, y: np.ndarray) -> float: # output type is float not np.ndarray
        if (self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None): # using "is" instead of "=="
            raise ValueError("Parameters not defined")
        m = A2.shape[0]
        A2_clipped = np.clip(A2, 1e-15, 1 - 1e-15)
        y = y.reshape(-1,1) # reshaped y to bring 1D y to 2D (m,) -> (m,1)
        cost = np.where(y == 1, -np.log(A2_clipped), -np.log(1 - A2_clipped)).mean()
        cost_reg = (np.sum(self.W1**2) + np.sum(self.W2**2))*self.lambda_/(2*m)
        return float(cost + cost_reg)
    def _backward(self, X: np.ndarray, y: np.ndarray, Z1: np.ndarray, A1: np.ndarray, A2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None): # using "is" instead of "=="
            raise ValueError("Parameters not defined")
        y = y.reshape(-1,1)
        m = X.shape[0]
        dZ2 = A2 - y #(m,1)
        dW2 = (A1.T @ dZ2) / m + self.W2 * (self.lambda_ / m)
        db2 = np.mean(dZ2, axis = 0, keepdims = True) # collapses (m,1) -> (1,1)
        dZ1 = (dZ2 @ self.W2.T) * np.where(Z1<=0, 0, 1)  # (m,1)*(1,h) -> (m,h) 
        dW1 = (X.T @ dZ1)/m + self.W1*(self.lambda_/m)
        db1 = np.mean(dZ1, axis = 0, keepdims = True) # collapses (m,h) -> (1,h)
        return (dW1, db1, dW2, db2)
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> TwoLayerNeuralNetwork:
        m,n = X.shape
        self._init_params(n)
        if (self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None): # using "is" instead of "=="
            raise ValueError("Parameters not defined")
        self.cost_history = []
        frequency = max(1, self.steps // 10)
        for i in range(self.steps):
            Z1, A1, Z2, A2 = self._forward(X)
            cost = self._compute_cost(A2, y)
            dW1, db1, dW2, db2 = self._backward(X, y, Z1, A1, A2)
            #Update Phase
            self.W1 -= self.alpha * dW1
            self.b1 -= self.alpha * db1
            self.W2 -= self.alpha * dW2
            self.b2 -= self.alpha * db2
            self.cost_history.append(cost)
            #Print Progress
            if verbose:
                if i == 0 or i + 1 == self.steps or (i+1) % frequency == 0 :
                    end_char = "\n"
                else:
                    end_char = "\r"
                print(f"Step {i+1}/{self.steps} | cost = {cost:.4f}", end = end_char, flush = True)
        return self
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        _, _, _, A2 = self._forward(X)
        A2 = np.where(A2 >= threshold, 1, 0)
        return A2.ravel()