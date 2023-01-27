# Import your model:
from model import LinearRegression, GradientDescentLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Tuple

model = GradientDescentLinearRegression()

# tiny dataset
alpha, beta = -2.0, 1.4
xsynth = np.linspace(-1, 1, 10).reshape(-1, 1)  # 100 examples, in 10D space
ysynth = (alpha * xsynth + beta).reshape(-1)  # target values
# xsynth_append = np.hstack((xsynth, np.ones((10, 1)))
#                           )  # Append a column with `1`

# uncomment the lines below to test the analytical solution
model.fit(xsynth, ysynth)
params = (model.w, model.b)
print(f"Original: alpha={alpha} and beta={beta}")
print(f"Estimated: alpha={params[0].item()} and beta={params[1].item()}")
