# Import your model:
from model import LinearRegression, GradientDescentLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Tuple

model = LinearRegression()

# tiny dataset
alpha, beta = np.array([10,33]), 1.4
xsynth = np.linspace(-10, 10, 100).reshape(-1, 2)  # 100 examples, in 10D space
ysynth = xsynth @ alpha + beta  # target values
# xsynth_append = np.hstack((xsynth, np.ones((10, 1)))
#                           )  # Append a column with `1`
print(xsynth)

# uncomment the lines below to test the analytical solution
model.fit(xsynth, ysynth)
params = (model.w, model.b)

print(f'Original: alpha={alpha} and beta={beta}')
print(f'Estimated: alpha={params[0]} and beta={params[1]}')

model.predict(xsynth[3])
