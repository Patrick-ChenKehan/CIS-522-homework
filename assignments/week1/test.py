# Import your model:
from model import LinearRegression, GradientDescentLinearRegression
import numpy as np

model = GradientDescentLinearRegression()
# model = GradientDescentLinearRegression()

# tiny dataset
alpha, beta = np.array([2, 3]), 7
xsynth = np.stack(
    (np.random.normal(0, 3, 100), np.random.normal(0, 5, 100))
).T  # 100 examples, in 10D space
ysynth = (xsynth @ alpha + beta).reshape(-1)  # target values
# xsynth_append = np.hstack((xsynth, np.ones((10, 1)))
#                           )  # Append a column with `1`
print(xsynth)
# uncomment the lines below to test the analytical solution
model.fit(xsynth, ysynth)
params = (model.w, model.b)
print(f"Original: alpha={alpha} and beta={beta}")
print(f"Estimated: alpha={params[0]} and beta={params[1]}")
