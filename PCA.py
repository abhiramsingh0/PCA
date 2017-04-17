import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#"initializations"
no_of_points = 1000

#"mean and variance"
mean = np.zeros(3)
cov = np.identity(3)

#"generating data points according to gaussian distribution"
x = np.random.multivariate_normal(mean, cov, no_of_points).T
print "dimension of matrix is: (%d, %d)" %x.shape

#"plotting data points generated from normal distribution"
ax.scatter(x[0,:], x[1,:], x[2,:], zdir='z', s=10)
#plt.show()
#plt.close(fig)"
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
print "selecting 2 maximum eigen values and eigen vectors"
evec_new = evec_norm[:,:2]
print "eigen vectors for projection are following"
print evec_new

print "---------------------------------------------------"
print "transforming the data to new dimensions"
x_new = np.dot(evec_new.T, x)
print "new data points dimesion (%d %d)"%x_new.shape



