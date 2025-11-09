import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=1000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # bias term
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            z = X.dot(self.theta)
            h = self.sigmoid(z)
            grad = X.T.dot(h - y) / len(y) + self.l2 * self.theta
            self.theta -= self.lr * grad
        return self

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
