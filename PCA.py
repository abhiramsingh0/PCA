import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"initializations"
no_of_points = 1000

"mean and variance"
mean = np.zeros(3)
cov = np.identity(3)

"generating data points according to gaussian distribution"
x = np.random.multivariate_normal(mean, cov, no_of_points)
print "dimension of data points is: (%d, %d)" %x.shape
