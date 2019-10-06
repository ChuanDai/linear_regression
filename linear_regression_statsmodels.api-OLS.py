# https://www.geeksforgeeks.org/ml-multiple-linear-regression-backward-elimination-technique/
import numpy as np
import statsmodels.api as sm

# for cross validation the linear regression results from other model

'''
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
'''


def generate_dataset(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i/2 + np.random.rand()*n
        x.append([1, x1, x2])
        y.append(random_x1 * x1 + random_x2 * x2 + 1)
    return np.array(x), np.array(y)


x, y = generate_dataset(200)

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
