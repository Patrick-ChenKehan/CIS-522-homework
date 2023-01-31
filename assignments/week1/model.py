"""
File for LinearRegression and GradientDescentLinearRegression
"""
import numpy as np


class LinearRegression:
    """
    A linear regression model that uses close-form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """
        Initialize attributes
        """
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fiting X and y with linear regression model

        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training targets

        Raises:
            RuntimeError: LinAlgError. Matrix is Singular. No analytical solution.
        """
        # Append 1 to X
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        if np.linalg.det(X.T @ X) != 0:
            params = np.linalg.inv(X.T @ X) @ X.T @ y
            self.w = params[:-1]
            self.b = params[-1]
        else:
            raise RuntimeError(
                "LinAlgError. Matrix is Singular. No analytical solution."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the output for the given input.

        Args:
            X (np.ndarray): Data to predict

        Raises:
            RuntimeError: Model not fitted

        Returns:
            np.ndarray: Prdiction of target
        """

        if self.w is None:
            raise RuntimeError("Model not fitted")

        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def _mse(self, target, input) -> float:
        return ((target - input) ** 2).mean()

    def _gradient_descent(
        self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, lr: float = 0.01
    ) -> None:
        # Compute the gradient
        N = X.shape[0]
        dw = (-2 / N) * np.dot(X.T, (y.reshape(-1) - y_hat))
        db = (-2 / N) * (y.reshape(-1) - y_hat).sum()

        # Update weight and bias
        self.w -= lr * dw
        self.b -= lr * db

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.001, epochs: int = 10000
    ) -> None:
        """Fit the linear regression model to X with gradient descent

        Args:
            X (np.ndarray): data to fit
            y (np.ndarray): target of data
            lr (float, optional): learning rate for GD. Defaults to 0.01.
            epochs (int, optional): epochs for GD. Defaults to 1000.
        """
        # Initialize w and b as tensor
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn(1)

        # Add log
        X = X - X.mean(axis=0)
        # Start training
        for _ in range(epochs):
            y_hat = self.predict(X)
            loss = self._mse(y, y_hat)
            # Apply GD
            self._gradient_descent(X, y, y_hat, lr)
        print(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Raises:
            RuntimeError: Model not fitted

        Returns:
            np.ndarray: The predicted output.

        """
        if self.w is None:
            raise RuntimeError("Model not fitted")

        # Compute prediction
        return X @ self.w + self.b
