import numpy as np

class LinearRegression:
    def __init__(self, alpha: float=0.001, num_steps: int = 1000, lambda_:float=0.0):
        self.alpha = alpha
        self.num_steps = num_steps
        self.lambda_=lambda_
        self.w = None
        self.b = None
        self.cost_history = []
    def predict(self, X: np.ndarray)->np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("The model must be fit with .fit() before calling .predict()")
        return np.dot(X, self.w) + self.b
    def _compute_cost(self, X, y)->float:
        m = X.shape[0]
        y_hat = self.predict(X)
        mse_cost = np.sum((y_hat-y)**2)/(2*m)
        if self.w is None or self.b is None:
                    raise ValueError("The model must be fit with .fit() before calling .predict()")
        reg_cost = (self.lambda_/(2*m))*np.sum(self.w**2)
        return mse_cost + reg_cost
    def _compute_gradient(self, X, y):
        m = X.shape[0]
        y_hat = self.predict(X)
        err = y_hat-y
        dJ_dw = np.dot(X.T,err)/m + (self.lambda_/m)*self.w
        dJ_db = np.sum(err)/m
        return dJ_dw, dJ_db
    def fit(self, X:np.ndarray, y:np.ndarray):
        m,n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.num_steps):
            dJ_dw, dJ_db = self._compute_gradient(X,y)
            self.w = self.w - self.alpha*dJ_dw
            self.b = self.b - self.alpha*dJ_db
            cost = self._compute_cost(X,y)
            self.cost_history.append(cost)

        return self