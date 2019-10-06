import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# randomly provided data set
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# concatenate an all ones vector as the first column
# for the coefficients of x_0
x = np.hstack((np.ones((len(y), 1)), x))

# x's transposition
x_t = x.transpose()

# dot multiply x's transposition and x
x_t_dot_x = np.dot(x_t, x)

# inverse the dot multiply of x's transposition and x
inv_x_t_dot_x = np.linalg.inv(x_t_dot_x)

# dot multiply x's transposition and y
x_t_dot_y = np.dot(x_t, y)

# get thetas
thetas = np.dot(inv_x_t_dot_x, x_t_dot_y)

# calculate the prediction of y
y_predict = np.dot(x, thetas)

# print the coefficients
print('The coefficients are:')
for i in range(len(thetas)):
    print(thetas[i])

# draw a chart
fig = plt.figure()
ax = fig.gca(projection='3d')

plt.scatter(x[:, 1], x[:, 2], y, label='y', color="dodgerblue")
plt.scatter(x[:, 2], x[:, 2], y_predict, label='regression', color="orange")

ax.legend()
plt.show()
