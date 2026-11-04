import numpy as np

class LogisticRegression:
    def __init__(self, alpha: float = 0.001, steps: int=1000, lambda_:float=0.0):
        self.alpha = alpha
        self.steps = steps
        self.lambda_=lambda_
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.cost_history: list[float] = []
    
    def _sigmoid(self,z:np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-z))
    def _compute_cost(self, X:np.ndarray, y:np.ndarray)-> float:
        m = X.shape[0]
        if self.w is None or self.b is None: 
            raise ValueError("Model is not fitted yet.")
        z = np.dot(X,self.w) + self.b
        y_hat = self._sigmoid(z)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        cost = np.where(y==1, -np.log(y_hat), -np.log(1-y_hat)).mean()
        cost_reg = (self.lambda_/(2*m))*np.dot(self.w,self.w)
        return float(cost+cost_reg)
    def _compute_gradient(self, X:np.ndarray, y:np.ndarray)->tuple[np.ndarray,float]:
        m = X.shape[0]
        if self.w is None or self.b is None: 
            raise ValueError("Model is not fitted yet.")
        z = np.dot(X,self.w) + self.b
        y_hat = self._sigmoid(z)
        dJ_dw = np.dot(X.T, (y_hat-y))/m + (self.lambda_/m)*self.w
        dJ_db = np.sum(y_hat-y)/m
        return dJ_dw, float(dJ_db)
    def fit(self, X:np.ndarray, y:np.ndarray, verbose:bool=False)->LogisticRegression:
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []        #reset cost history
        for i in range(self.steps):
            dJ_dw, dJ_db = self._compute_gradient(X,y)
            self.w = self.w - self.alpha*dJ_dw
            self.b = self.b - self.alpha*dJ_db
            cost = self._compute_cost(X,y)
            self.cost_history.append(cost)
            frequency = max(1, self.steps // 10)
            if verbose:
                if (i+1)%frequency==0 or i==0 or i+1 == self.steps:
                    end_char = "\n"
                else:
                    end_char = "\r"
                print(f"Step {i+1}/{self.steps} | cost:{cost:.4f}", end = end_char, flush = True) 
        return self
    def predict(self, X:np.ndarray, threshold:float=0.5)->np.ndarray:
        y_hat = self._sigmoid(X@self.w + self.b)
        return np.where(y_hat>=threshold, 1, 0)
    
    
        
    