# https://gist.github.com/samueljackson92/8148506
# http://cs229.stanford.edu/notes/cs229-notes1.pdf
# http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy-machine-learning
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(self, alpha=0.001, tolerance=0.02, max_iterations=500):
        # alpha is the learning rate or size of step to take in
        # the gradient decent
        self._alpha = alpha
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        # thetas is the array coefficients for each term
        # the last element is the y-intercept
        self._thetas = None

    def fit(self, xs, ys):
        num_examples, num_features = np.shape(xs)
        self._thetas = np.ones(num_features)

        xs_transposed = xs.transpose()
        for i in range(self._max_iterations):
            # difference between our hypothesis and actual values
            diffs = np.dot(xs, self._thetas) - ys
            # sum of the squares
            cost = np.sum(diffs ** 2) / (2 * num_examples)
            # calculate average gradient for every example
            gradient = np.dot(xs_transposed, diffs) / num_examples
            # update the coefficients
            self._thetas = self._thetas - self._alpha * gradient
            # check if this fit is "good enough"
            if cost < self._tolerance:
                return self._thetas

        return self._thetas

    def predict(self, x):
        return np.dot(x, self._thetas)


# randomly provided data set
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# concatenate an all ones vector as the first column
# for the coefficients of x_0
x = np.hstack((np.ones((len(y), 1)), x))

gd = GradientDescent(tolerance=0.001)
thetas = gd.fit(x, y)

print('The coefficients are:')
for i in range(len(thetas)):
    print(thetas[i])

y_predict = gd.predict(x)

# draw a chart
fig = plt.figure()
ax = fig.gca(projection='3d')

plt.scatter(x[:, 1], x[:, 2], y, label='y', color="dodgerblue")
plt.scatter(x[:, 2], x[:, 2], y_predict, label='regression', color="orange")

ax.legend()
plt.show()
