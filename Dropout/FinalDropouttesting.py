""""
Created on Thu Feb 20 21:11:27 2020

@author: Yarin Gal
"""
#%%% Imports
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import math
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
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
import time

#%%%% Gall's class
class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1, dropout = 0.01):

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

        inputs = Input(shape=(1))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=keras.regularizers.l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=keras.regularizers.l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(1, kernel_regularizer=keras.regularizers.l2(reg))(inter)
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
    
    
#%%% Data generation
def y(x):
    return 0.5 * (x ** 2)

def sigma(x):
    return 0.3 * np.exp(x ** 2)

def get_data(N_train, N_test):
    
    "Create a dataset containing of N_train training samples"
    "and N_test testing samples genereated according to y(x)"
    "with an added noise term with variance sigma^2"
    
    Xtrain=np.array(np.linspace(-1,1,N_train)) 
    Ytrain=np.zeros(N_train)
    Xtest=np.array(np.linspace(-0.99,0.99,N_test))
    Ytest=np.zeros(N_test)
    
    for i in range(0,N_train):
        Ytrain[i]=y(Xtrain[i])+np.random.normal(0,sigma(Xtrain[i]))
    
    for i in range(0, N_test):
        Ytest[i]=y(Xtest[i])+np.random.normal(0,sigma(Xtest[i]))
        
    return Xtrain, Ytrain, Xtest, Ytest

Xtrain, Ytrain, Xtest, Ytest = get_data(10000, 2000)
plt.plot(Xtrain,Ytrain)
plt.plot(Xtrain,y(Xtrain))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test data')
plt.show()

#%%% Initiate and train model using gall
model = net(Xtrain, Ytrain, n_hidden = np.array([50, 50, 50])).model
predictions = model.predict(Xtest)
for i in range(0, 100):
    predictions += model.predict(Xtest)

plt.plot(Xtest, predictions/101, label = 'predicted')
plt.plot(Xtest, y(Xtest), label = 'real')
plt.legend()
plt.title('real and predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.plot(Xtest,Ytest)
plt.plot(Xtest,y(Xtest))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test data 2')
plt.show()

#%%% Test 1
def test(x, M, T):
    
    "ADD docstring"
    
    testresults1 = np.zeros(M)
    testresults2 = np.zeros(M)
    for i in range(0, M):
        Xtrain, Ytrain, Xtest, Ytest = get_data(10000, 2000)
        model = net(Xtrain, Ytrain, n_hidden = np.array([50, 50, 50])).model
        predictions = np.zeros(T)
        for j in range(0, T):
            predictions[j] = model.predict(np.array([x]))
        testresults1[i] = (np.mean(predictions) - y(x)) / np.std(predictions)
        testresults2[i] = len(predictions[predictions < y(x)]) / T
    return testresults1, testresults2


xtests = np.linspace(-1, 1, 5)
for x in xtests:
    test_1, test_2 = test(x, 100, 200)
    plt.title("x = {x}, sigma = {std}".format(x = x, std = round(np.std(test_1),2)))
    plt.hist(test_1)
    plt.show()
    plt.title("x = {x}".format(x = x))
    plt.hist(test_2)
    plt.show()
    
    

