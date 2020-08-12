#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:24:01 2020

@author: Laurens Sluijterman (the class is written by Yarin Gall)
"""
#%%% Imports
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from scipy.special import logsumexp
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import keras
import tensorflow as tf
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
import time
import random

#%%%% Gall's class
class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1, dropout = 0.05):

        """ Author: Yarin Gall
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            pass
        else:
            pass

        
        # We construct the network
        N = len(X_train)
        batch_size = 128
        lengthscale = 1
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(15))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=keras.regularizers.l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=keras.regularizers.l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(2, kernel_regularizer=keras.regularizers.l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')

        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0)
        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!
    
        self.model = model

    def predicts(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

        T = 1000
        
        Yt_hat = np.array([model.predict(X_test, batch_size=500, verbose=0) for _ in range(T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)

        # We are done!
        return model
    
def soft(x):
    """Return log(1 + exp(x)) + 1e-6."""
    return 1e-6 + tf.math.softplus(x)

def y_difficult_1(x):
    """Return the mean as function of x."""
    return 0.1 * x[0] + 0.3 * x[1] + 0.1 * x[7] - x[8] * x[11]

def y_difficult_2(x):
    """Return the mean as function of x."""
    return 0.1 * x[3] + 0.3 * x[1] + 0.1 * x[8] - x[11] * x[7]

def sigma_difficult(x):
    """Return the standard deviation as a function of x."""
    return 0.05 * soft(0.1 * x[1] + 0.2 * x[2] - 0.05 *  x[4] * x[5] + 0.1 * x[11]
                - 0.5 * x[12] + 0.3 * x[14] + 0.2 * x[8])


def get_fancy_data(N_train, N_test, mean, A):
    """
    Get a difficult dataset.
    
    This function gets a dataset consisting of N_train training values
    and N_test test values. The y values are 1-dimensional and the x-values
    are dim(mean) dimensional. The matrix A specifies the covariance matrix
    of the noise (Sigma = A^T dot A). First, an x value is sampeled from
    N(mean, Sigma). Secondly, y is calculated via "y_difficult" and 
    normally distriubted noise is added with standard deviation 
    "sigma_difficult(x)".

    Parameters:
        N_train (int): The number of desired training samples.
        N_test (int): The number of desired test samples.
        mean: (d x1 matrix) The mean of the normal distribution from 
            which x is sampled.
        A (d x d matrix): A matrix that specifies the covariance matrix for
            the normal distribution from which x is sampeled. 

    Returns:
        X_train: An array of size N_train by dim(mean) consisting of the 
                training x-values.
        Y_train: An array of size N_train by 1 consisting of the y values 
                corresponding to X_train.
        X_test: An array of size N_testby dim(mean) consisting of the 
                training x-values.
        Y_est: An array of size N_test by 1 consisting of the y values 
                corresponding to X_test.
                
    """
    cov = np.transpose(A).dot(A)
    X_train = np.random.multivariate_normal(mean, cov, (N_train,))
    Y_train = np.zeros((N_train, 2))
    X_test = np.random.multivariate_normal(mean, cov, (N_test,))
    Y_test = np.zeros((N_test, 2))
    for i in range(N_train):
        Y_train[i,0] = y_difficult_1(X_train[i]) \
            + np.random.normal(0, sigma_difficult(X_train[i]))
        Y_train[i,1] = y_difficult_2(X_train[i]) \
            + np.random.normal(0, sigma_difficult(X_train[i]))            
    for i in range(N_test):
        Y_test[i,0] = y_difficult_1(X_test[i]) + \
                  np.random.normal(0, sigma_difficult(X_test[i]))
        Y_test[i,1] = y_difficult_1(X_test[i]) + \
                  np.random.normal(0, sigma_difficult(X_test[i]))
    return X_train, Y_train, X_test, Y_test 



#%% Testing 
np.random.seed(2)
mean = [0, 0.1, 0.2, 0.3, 0.1, 
        0.5, -0.3, 0.1, -0.2, 0.5,
        0.5, 0, 0.2, 0.2, -0.2]
A = [[0.1,  0.4, 0, 0, 0,  0,  0, 0.1, 0.2, -0.3, 0.2, 0.1, -0.1, 0.1, 0], 
      [0.1, 0.3, 0, 0, 0,  0.1, 0, 0.1, -0.3, -0.3, 0, 0, 0.1, -0.2, 0.3],
      [0,    0, 0,   0, 0, 0, -0.1, 0.1, 0.4, -0.3, 0.1, 0, 0, 0, 0],
      [0.4,  0, 0,  0.2, 0, 0,  0.21, 0, 0, 0, 0, 0.1, 0.4, 0.1, 0.2],
      [0,    0, 0,   0,   0.4, 0, 0.2, 0, -0.2, 0.2, -0.1, 0.1, 0.3, 0.1, 0],
      [-0.2,  0, 0, -0.2, 0,   0.1, 0, -0.2,  0, 0, -0.2, 0,   0.1, 0, 0],
      [0,    0, 0.7, 0, 0, 0, 0.2, 0,    0, 0.7, 0, 0, 0, 0.2,0.1 ]]

# Testing an individual run
X_train, Y_train, X_test, Y_test = get_fancy_data(20000, 1000, mean, A)  

real_y_test = np.zeros((1000, 2))
for i in range(1000):
    real_y_test[i, 0] = y_difficult_1(X_test[i])
    real_y_test[i, 1] = y_difficult_2(X_test[i])


#Train a single model and evaluate the results
T_1 = np.array(np.zeros(1000))
T_2 = np.array(np.zeros(1000))
for j in range(0, 100):
    X_train, Y_train = get_fancy_data(20000, 0, mean, A)[0],  \
                       get_fancy_data(20000, 0, mean, A)[1]
    model = net(X_train, Y_train, n_hidden = np.array([50, 50, 50])).model
    predictions = [model.predict(X_test)]
    for i in range(0, 99):
        predictions.append(model.predict(X_test))
    T_1 = np.vstack(((np.mean(predictions, axis = 0)[:,0] - real_y_test[:,0]) /\
               np.std(predictions, axis = 0)[:,0], T_1))
    T_2 = np.vstack(((np.mean(predictions, axis = 0)[:,1] - real_y_test[:,1]) /\
               np.std(predictions, axis = 0)[:,1], T_2))
    print(j, '/100')
    
    

plt.hist(np.std(T_1[:100], axis = 0), label = 'T_{1}')
plt.title("Standard deviation of $T_{1}$")
plt.xlabel("std(T)")
plt.show()


plt.hist(np.std(T_2[:100], axis = 0), label = 'T_{2}', bins = 10)
plt.title("Standard deviation of $T_{2}$")
plt.xlabel("std(T)")
plt.show()


