""""
Created on Thu Feb 20 21:11:27 2020

@author: Laurens Sluijterman (the class is written by Yarin Gall)
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
    

def y(x):
    """Return the mean as function of x."""
    return 0.5 * (x ** 2)

def sigma(x):
    """Return the standard deviation as a function of x."""
    return 0.3 * np.exp(x ** 2)

def get_data(N_train, N_test):    
    """
    Create a dataset.
    
    This function reates a dataset containing of N_train training samples
    and N_test testing samples genereated according to y(x)
    with an added noise term with variance sigma^2.
    
    Parameters:
        N_train (int): The number of training samples 
        N_test  (int): The number of test samples
    
    Returns:
        X_train, Y_train, X_test, Y_test: arrays genereated using y(x) as the mean
        and a normal noise with standar deviation sigma(x).
        
    """    
    Xtrain = np.array(np.linspace(-1,1,N_train)) 
    Ytrain = np.zeros(N_train)
    Xtest = np.array(np.linspace(-0.999,0.999,N_test)) 
    Ytest = np.zeros(N_test)
    
    for i in range(0,N_train):
        Ytrain[i]=y(Xtrain[i]) + np.random.normal(0,sigma(Xtrain[i]))
    
    for i in range(0, N_test):
        Ytest[i]=y(Xtest[i]) + np.random.normal(0,sigma(Xtest[i]))
        
    return Xtrain, Ytrain, Xtest, Ytest


def test(x, M, T):   
    """
    Test if MC-dropout works.
    
    This function implements two tests on MC-dropout. We make M datasets
    containing 10.000 datapoints. With this dataset, the model is trained. 
    mu_hat is calculated using T forward passes trough the network and taking 
    the mean. The predicted uncertainty is obtained by taking the standard
    deviation. The first is to evaluate mu_hat(x) - mu(x) / std(predictions).
    The second test is to look at the fraction of forward passes that is 
    smaller than the actual value of mu. 
    
    Parameters:
        x: The x value at which the tests are evaluated.
        M: The number of simulations.
        T: The number of forward passes.
        
    Returns:
        testresults1: An array containing the M evaluations of test 1
        testresults2: An array containing the M evaluations of test 2
    
    """ 
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


#%% Testing and visualization

#Visualize the test data
Xtrain, Ytrain, Xtest, Ytest = get_data(10000, 2000)
plt.plot(Xtrain,Ytrain)
plt.plot(Xtrain,y(Xtrain))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test data')
plt.show()

#Train a single model and evaluate the results
model = net(Xtrain, Ytrain, n_hidden = np.array([200, 100, 50])).model
predictions = [model.predict(Xtest)]
for i in range(0, 100):
    predictions.append(model.predict(Xtest))

plt.errorbar(Xtest, np.mean(predictions, axis = 0), yerr = np.std(predictions,
             axis = 0), errorevery = 50, label = 'predicted')
plt.plot(Xtest, y(Xtest), label = 'real')
plt.legend()
plt.title('real and predicted, p = 0.05')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Run the tests for multiple simulations and multiple values of x.
#Note that this code is inefficient since we could use 1 simulation to test
# for all 5 values of x. For our purposes this does not really matter. 

xtests = np.linspace(-1, 1, 5)
for x in xtests:
    test_1, test_2 = test(x, 100, 200)
    plt.title("x = {x}, sigma = {std}".format(x = x, std = round(np.std(test_1),2)))
    plt.hist(test_1)
    plt.show()
    plt.title("x = {x}".format(x = x))
    plt.hist(test_2)
    plt.show()
    
    

