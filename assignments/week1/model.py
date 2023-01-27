import numpy as np
import torch

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

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Fiting X and y with linear regression model

        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training targets

        Raises:
            RuntimeError: LinAlgError. Matrix is Singular. No analytical solution.
        """
        # Append 1 to X
        X = np.vstack((X, np.ones((1, X.shape[1])))).T

        if np.linalg.det(X.T @ X) != 0:
            params = np.linalg.inv(X.T @ X) @ X.T @ y
            # print(self.params)
            self.w = params[:-1]
            self.b = params[-1]
        else:
            # print("LinAlgError. Matrix is Singular. No analytical solution.")
            raise RuntimeError(
                "LinAlgError. Matrix is Singular. No analytical solution.")
            
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the output for the given input.

        Args:
            X (np.ndarray): Data to predict

        Raises:
            RuntimeError: Model not fitted

        Returns:
            np.ndarray: Prdiction of target
        """
        
        if not self.w:
            raise RuntimeError("Model not fitted")
        
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """
    
    def _gradient_descent(self, lr: float = 0.01) -> None:
        with torch.no_grad():
            self.w -= self.w.grad * lr
            self.b -= self.b.grad * lr
            
            # Set gradient to zero
            self.w.grad.zero_()
            self.b.grad.zero_()

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """Fit the linear regression model to X with gradient descent

        Args:
            X (np.ndarray): data to fit
            y (np.ndarray): target of data
            lr (float, optional): learning rate for GD. Defaults to 0.01.
            epochs (int, optional): epochs for GD. Defaults to 1000.
        """
        # Convert data to tensor
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        
        # Initialize w and b as tensor
        self.w = torch.randn(X.shape[1], requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        
        # Set criteria for GD
        criteria = torch.nn.MSELoss()
        
        # Start training
        for epoch in range(epochs):
            y_hat = self.predict(X)
            loss = criteria(y, y_hat)
            loss.backward()
            self._gradient_descent(lr)
            
        self.w = self.w.detach().numpy()
        self.b = self.b.detach().numpy()
    

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

        return X @ self.w + self.b
