# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:39:22 2016

@author: qwertzuiopu
"""

import numpy as np
import scipy.io as sio

def usingSVD(dataMatrix, desiredVariancePercentage=1.0):
    # This function should implement the PCA using the Singular Value
    # Decomposition (SVD) of the given dataMatrix
    
    # De-Meaning the feature space
    meanDataMatrix = dataMatrix.mean(1)
    demeanedDataMatrix = (dataMatrix.T - meanDataMatrix)
    
    # SVD Decomposition
    # You need to transpose the data matrix
    U, s, V = np.linalg.svd(demeanedDataMatrix / np.sqrt(dataMatrix.shape[1]-1)) # Divide by sqrt(N-1) to get consistent results
    V = V.T     # Numpy gives V transposed
    
    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.
    for i in range(V.shape[0]):
        if V[:,i][np.argmax(np.fabs(V[:,i]))] < 0:
            V[:,i] = -V[:,i]
            
    # Compute the accumelative Eigenvalues to finde the desired
    # Variance
    s = s**2
    acc_eigenvals = np.sum(s)
    
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance.
    for i in range(s.shape[0]):
        if np.sum(s[:i+1])/acc_eigenvals >= desiredVariancePercentage:
            eigenvals = s[:i+1]
            eigenvecs = V[:,:i+1]
            break
    
    # Project the data
    projectedData = np.dot(demeanedDataMatrix, eigenvecs)
    
    # Return Data
    return eigenvecs, eigenvals, meanDataMatrix, demeanedDataMatrix.T, projectedData.T
 
def usingCOV(dataMatrix, desiredVariancePercentage=1.0):
    # This function should implement the PCA using the
    # EigenValue Decomposition of a given Covariance Matrix 
    
    # De-Meaning the feature space 
    meanDataMatrix = dataMatrix.mean(1)
    demeanedDataMatrix = (dataMatrix.T - meanDataMatrix)
            
    # Computing the Covariance 
    covMatrix = np.dot(demeanedDataMatrix.T, demeanedDataMatrix)
            
    # Eigen Value Decomposition
    # In COV, you need to order the eigevectors according to largest eigenvalues
    eigenValues, eigenVectors = np.linalg.eig(covMatrix / (dataMatrix.shape[1] - 1))
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.
    for i in range(eigenVectors.shape[0]):
        if eigenVectors[:,i][np.argmax(np.fabs(eigenVectors[:,i]))] < 0:
            eigenVectors[:,i] = -eigenVectors[:,i]


    # Compute the accumelative Eigenvalues to finde the desired
    # Variance 
    acc_eigenvals = np.sum(eigenValues)
            
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance. 
    for i in range(eigenValues.shape[0]):
        if np.sum(eigenValues[:i+1])/acc_eigenvals >= desiredVariancePercentage:
            eigenvals = eigenValues[:i+1]
            eigenvecs = eigenVectors[:,:i+1]
            break
            
    # Project the data 
    projectedData = np.dot(demeanedDataMatrix, eigenvecs)
    
    # Return data
    return eigenvecs, eigenvals, meanDataMatrix, demeanedDataMatrix.T, projectedData.T