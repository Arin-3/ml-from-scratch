import numpy as np

class DeepNeuralNetwork():
    def __init__(self, hidden_layers: tuple[int, ...] = (16,16), alpha: float = 0.01, steps: int = 1000, lambda_: float = 0.0, random_state: int | None = None) -> None:
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.steps = steps
        self.lambda_ = lambda_
        self.random_state = random_state
        
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.cost_history: list[float] = []
    
    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights = []
        self.biases = []
        layer_dims = [n_features, *self.hidden_layers, 1] #using iterable unpacking
        for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:]): #layer_dims[:-1] are the input length and layer_dims[1:] are the output size
            self.weights.append(rng.standard_normal((n_in, n_out)) * np.sqrt(2 / n_in)) # He initialization
            self.biases.append(np.zeros((1, n_out)))
            
    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))
    
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.weights or not self.biases: 
            raise ValueError("Parameters not defined")
        Z_cache: list[np.ndarray] = []
        A_cache: list[np.ndarray] = [X]
        L = len(self.weights)
        for l in range(L):
            Z = A_cache[-1] @ self.weights[l] + self.biases[l]
            if l < L - 1:
                A = self._relu(Z)
            else:
                A = self._sigmoid(Z)
            Z_cache.append(Z)
            A_cache.append(A)
        return (Z_cache, A_cache)
    
    def _compute_cost(self, AL: np.ndarray, y: np.ndarray) -> float:
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        y = y.reshape(-1, 1)
        m = AL.shape[0]
        cost = np.where(y == 1, -np.log(AL), -np.log(1 - AL)).mean()
        cost_reg = sum(np.sum(W ** 2) for W in self.weights) * self.lambda_ / (2*m)
        return float(cost + cost_reg)
    
    def _backward(self, y: np.ndarray, Z_cache: list[np.ndarray], A_cache: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]] :
        if not self.weights or not self.biases:
            raise ValueError("Parameters not defined")
        y = y.reshape(-1, 1)
        m = y.shape[0]
        dW_list: list[np.ndarray] = []
        db_list: list[np.ndarray] = []
        
        dZ = A_cache[-1] - y # dZ of layer L-1(Last Layer)
        
        L = len(self.weights)
        for l in range(L-1, -1, -1): # Loop from L-1 to 0
            dW = (A_cache[l].T @ dZ) / m + self.weights[l] * (self.lambda_/m)
            db = np.mean(dZ, axis = 0, keepdims = True) 
            if l > 0:
                dZ_prev = (dZ @ self.weights[l].T) * np.where(Z_cache[l-1] <= 0, 0, 1) #dZ of previous layer (l-1)
                dZ = dZ_prev
                
            #Update Phase
            dW_list.append(dW)
            db_list.append(db)
            
        dW_list.reverse()
        db_list.reverse()
        return (dW_list, db_list)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> DeepNeuralNetwork:
        m,n = X.shape
        self._init_params(n)
        self.cost_history = []
        frequency = max(1, self.steps // 10)
        
        for step in range(self.steps):
            Z_cache, A_cache = self._forward(X)
            cost = self._compute_cost(A_cache[-1], y)
            dW_list, db_list = self._backward(y, Z_cache, A_cache)
            
            #Update Phase
            for i in range(len(self.weights)):
                self.weights[i] -= self.alpha * dW_list[i]
                self.biases[i] -= self.alpha * db_list[i]
            self.cost_history.append(cost)
            
            if verbose:
                if step == 0 or (step + 1) % frequency == 0 or step + 1 == self.steps:
                    end_char = "\n"
                else:
                    end_char = "\r"
                print(f"Step {step+1}/{self.steps} | cost = {cost:.4f}", end = end_char, flush = True)
        return self
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        _, A_cache = self._forward(X)
        AL = A_cache[-1]
        return np.where(AL >= threshold, 1, 0).ravel()