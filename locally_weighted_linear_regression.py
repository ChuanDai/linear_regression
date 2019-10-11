# https://xavierbourretsicotte.github.io/loess.html
# https://gist.github.com/agramfort/850437
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


# Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi, x0, tau=.005):
    return np.exp(-(xi - x0)**2/(2*tau))


def lowess_bell_shape_kern(x, y, tau=.005):
    """
    lowess_bell_shape_kern(x, y, tau=.005) -> yest
    Locally weighted regression:
    fits a non-parametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements;
    each pair (x[i], y[i]) defines a data point in the scatterplot.
    The function returns the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau.
    Larger tau will result in a smoother curve.
    """
    n = len(x)
    yest = np.zeros(n)

    # Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])
    
    # Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest


# Initializing noisy non linear data
x = np.linspace(0, 1, 100)
noise = np.random.normal(loc=0, scale=.25, size=100)
y = np.sin(x * 1.5 * np.pi)
y_noise = y + noise


yest_bell = lowess_bell_shape_kern(x, y)


plt.figure(figsize=(10, 6))
plt.plot(x, y, color='darkblue', label='sin()')
plt.scatter(x, y_noise, facecolors='none', edgecolor='darkblue', label='sin() + noise')
plt.fill(x[:40], kernel_function(x[:40], 0.2, .005), color='lime', alpha=.5, label='Bell shape kernel')
plt.plot(x, yest_bell, color='red', label='Loess: bell shape kernel')
plt.legend()
plt.title('Sine with noise: Loess regression and bell shaped kernel')
plt.show()
