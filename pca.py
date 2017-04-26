import numpy as np
from numpy import linalg as LA

class PCA:
  # creating data members accessible to all objects
  # ...

  # creating object private data members
  def __init__(self, data_matrix):
    self.data = data_matrix
    # set each row as one sample
    self.data = self.set_sample_placement()
    # find mean of each component
    self.mean = self.find_mean()
    # find standard deviation of each component
    self.std = self.find_std()
    # now each component has 0 mean 1 variance
    self.norm_data = self.normalize()
    # find covariance matrix for normalized data
    self.cov_mat = self.find_cov()
    # find eigne value and eigen vectors for this cov_mat.
    self.evalue, self.evector = self.find_eval_evec()
    # sorted eigen values and corresponding eigen vectors
    self.s_evalue, self.s_evector = self.sort_eval_evec()

  # check if samples are in rows or columns
  def set_sample_placement(self, data_matrix):
    val = int(input("Enter 0 if samples are place row-wise\
        in matrix, 1 otherwise:"))
    if 1 == val :
      self.data = data_matrix.T
  
  # find mean of data points along each column
  def find_mean(self):
    return np.mean(self.data)

  # find mean of data points along each column
  def find_std(self):
    return np.std(self.data)

  # normalize data points along each component
  def normalize(self):
    return ((self.data - self.mean) / self.std)

  # find covarinace matrix of the normalized data points
  def find_cov(self):
    return np.cov(self.norm_data, rowvar = 0, bias = 1)

  # find eigne value and eigen vectors for this cov_mat.
  def find_eval_evec(self):
    return (LA.eig(cov_mat))

  # sorting eigen values and eigen vectors increasing order
  def sort_eval_evec(self):
    # find index after sorting of eigen values
    index = np.argsort(self.evalue)
    evalue = np.sort(self.evalue)
    # sort eigen vectors according to eigenvalues
    evec = self.evec[:,index]
    return (evalue, evec)

  # transforming data to the new dimension
  def transform_data(self, no_of_components):
    # extract top k eigen values
    evalu, evec = self.max_n_eval_evec(no_of_components)
    # calculate new data points
    new_components = self.norm_data * evec
    return (new_components)

  # select top k eigen values and eigen vectors
  def max_n_eval_evec(self, no_of_components):
    # find max n eigen values
    evalu = self.s_evalue[-no_of_components:]
    # find max n eigen vectors
    evec = self.s_evector[:, -no_of_components:]
    return (evalu, evec)
