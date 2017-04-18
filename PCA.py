import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes()

#"initializations"
no_of_points = 1000

#"mean and variance"
mean = np.zeros(2)
cov = np.identity(2)
cov[0][1] = -4
#cov[1][1] = 8
#"generating data points according to gaussian distribution"
x = np.random.multivariate_normal(mean, cov, no_of_points).T
print "dimension of matrix is: (%d, %d)" %x.shape

#"plotting data points generated from normal distribution"
plt.plot(x[0,:], x[1,:], 'ro')
print "----------------------------------------------------"
#"computing mean of data points"
mu = np.mean(x,axis=1)
print 'mean of data points along each component is: \n ', mu
sigma = np.cov(x)
print "covariance matrix is:\n", sigma

# computing eigen values and eigen vectors of covariance..
# ... matrix.

print "---------------------------------------------------"
evalue, evec = LA.eig(sigma)
print "eigen values are:\n", evalue
print "eigen vectors are:\n", evec

print "---------------------------------------------------"
index = np.argsort(evalue)
evalue = np.sort(evalue)
evec = evec[:,index]
print "sorted eigen values and eigen vectors"
print evalue
print "sorted eigen vectors according to eigen values"
print evec

print "---------------------------------------------------"
print "normalizing eigen vectors to have unit magnitude"
mag = LA.norm(evec, axis = 0)
evec_norm = evec/mag
print "normalized eigen vectors"
print evec

print "---------------------------------------------------"
print "selecting  maximum eigen values and eigen vectors"
evec_new = evec_norm[:,1]
print "eigen vectors for projection are following"
print evec_new[0], evec_new[1]
ax.arrow(0, 0, 6*evec_new[0],6*evec_new[1], \
		head_width=0.05, head_length=0.1)
print "---------------------------------------------------"
print "transforming the data to new dimensions"
x_new = np.dot(evec_new.T, x)
print "new data points dimesion"
print x_new.shape

print "---------------------------------------------------"
print "plotting data points along pricipal components"
#plt.plot(x[0,:], 'go')
plt.show()
