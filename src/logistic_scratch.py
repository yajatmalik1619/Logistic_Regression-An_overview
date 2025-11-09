import numpy as np

class LogisticRegressionScratch:

    def __init__(self, lr=0.01, n_iter=1000, l2=0.0, fit_intercept=True, tol=1e-6, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        X = self._add_intercept(X)
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)

        for i in range(self.n_iter):
            z = X.dot(self.theta)
            h = self._sigmoid(z)
            error = h - y
            grad = (X.T.dot(error) / n_samples) + (self.l2 * self.theta / n_samples)
            update = self.lr * grad
            self.theta -= update

            if self.tol and np.linalg.norm(update) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}")
                break

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
