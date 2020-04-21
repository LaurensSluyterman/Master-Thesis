#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:20:41 2020

@author: Laurens Sluijterman
"the structure of defining a neural network in a class was adapted from
 Yarin Gall's Github"
"""

#%% Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import gamma
import tensorflow.keras.backend as K
#%%
def y(x):  
    """ Returns the mean as function of x. """
    return 0.5 * (x ** 2)

def sigma(x):
    """" Returns the standard deviation as a function of x."""
    return 0.3 * np.exp(x ** 2)

def get_data(N_train, N_test):
    
    """
    Create a dataset containing of N_train training samples
    and N_test testing samples genereated according to y(x)
    with an added noise term with variance sigma^2
    
    Parameters:
        N_train (int): The number of training samples 
        N_test  (int): The number of test samples
    
    Returns:
        X_train, Y_train, X_test, Y_test: arrays genereated using y(x) as the mean
        and a normal noise with standar deviation sigma(x).
    """  
    X_train=np.array(np.linspace(-1,1,N_train)) 
    Y_train=np.zeros(N_train)
    X_test=np.array(np.linspace(-0.999,0.999,N_test))
    Y_test=np.zeros(N_test)
    for i in range(0,N_train):
        Y_train[i]=y(X_train[i])+np.random.normal(0,sigma(X_train[i]))
    for i in range(0, N_test):
        Y_test[i]=y(X_test[i])+np.random.normal(0,sigma(X_test[i]))
    return X_train, Y_train, X_test, Y_test

def soft(x):
    """Returns log(1 + exp(x)) + 1e-5"""
    return 1e-5 + tf.math.softplus(x)

class Neural_network: 
    
    """
   This class represents a model.
   
   Parameters:
       X_train: A matrix containing the inputs of the training data.
       Y_train: A matrix containing the targets of the training data
       n_hidden (array): An array containing the number of hidden units
               for each hidden layer. The length of this array 
               specifies the number of hidden layers.
       n_epochs: The number of training epochs for the main neural network.
       uncertainty_estimatios (boolean): If True the data will be split in two
               parts where the second part will be used to train a second
               neural network that will be used to give uncertainty estimates
               for both the mean and variance predictions of the first network.
        
   
    """
    
    def __init__(self, X_train, Y_train, n_hidden, n_epochs = 20,
                 uncertainty_estimates = True, n_hidden_2 = np.array([50]),
                 n_epochs_2 = 0, split = 0.2, verbose = True):
        if n_epochs_2 == 0:
            n_epochs_2 = n_epochs
        if uncertainty_estimates == True:
            X_train, X_train_2, Y_train, Y_train_2 = train_test_split(
                    X_train, Y_train, test_size = split)
        
        def loss_0(y, musigma):
            dist = tfd.Normal(loc = musigma[..., :1], 
                              scale = soft(musigma[...,1:]))
            return -np.sum(dist.log_prob(y))
        
        def loss(targets, outputs):
            mu = outputs[...,0:1]
            sigma = soft(outputs[...,1:2])
            y = targets[...,0:1]
            return - K.sum(- K.log(sigma) - 0.5 * K.square((y - mu) / sigma))
        
        c = 1 / (len(X_train))
        inputs = Input(shape=(1))
        inter = Dense(n_hidden[0], activation='relu', 
                      kernel_regularizer=keras.regularizers.l2(c))(inputs)
        for i in range(len(n_hidden) - 1):
            inter = Dense(n_hidden[i+1], activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(c))(inter)
        outputs = Dense(2, kernel_regularizer=keras.regularizers.l2(c), 
                        activation = 'linear')(inter)
        model = Model(inputs, outputs)
        
        model.compile(loss = loss, optimizer='adam')
        model.fit(X_train, Y_train, batch_size = len(X_train), epochs = n_epochs, 
                  verbose = verbose)
        self.model = model      
        
        if uncertainty_estimates == True: 
            verbose2 = verbose
            mu_hat = model.predict(X_train_2)[:,0]
            sigma_hat =  soft(model.predict(X_train_2)[:,1])
            targets = np.stack((Y_train_2, mu_hat, sigma_hat), axis = 1)
            
            def loss_1(targets, outputs):
                dist_1 = tfd.Normal(loc = outputs[..., :1], 
                                    scale = soft(outputs[...,1:2]))
                return -(dist_1.log_prob(targets[...,:1]))
            
            def loss_2(targets, outputs):
                dist_2 = tfd.Normal(loc = outputs[..., 1:2], 
                                    scale = soft(outputs[...,2:3]))
                return -(dist_2.log_prob(targets[...,1:2]))
            
            def loss_3(targets, outputs):
                sigma = soft(outputs[...,1:2])
                nu = soft(outputs[...,3:4])
                concentration = nu /2
                rate = 1 / (2 * sigma ** 2 / nu)
                dist_3 = tfd.Gamma(concentration = concentration, 
                                   rate = rate)
                return -(dist_3.log_prob(targets[..., 2:3] ** 2))
            
            def loss_4(targets, outputs):
                sigma = soft(outputs[...,1:2])
                nu = soft(outputs[...,3:4])
                dist_4 = tfd.Chi2(nu)
                return -(dist_4.log_prob(nu * targets[...,2:3] ** 2 / (sigma ** 2)))
            
            def loss_5(targets, outputs):
                sigma = soft(outputs[...,1:2])
                nu = soft(outputs[...,3:4])
                concentration = nu / 2
                rate = nu / 2
                dist_4 = tfd.Gamma(concentration = concentration, 
                                   rate = rate)
                return -(dist_4.log_prob(targets[...,2:3] ** 2 / (sigma ** 2)))

            def explicit_loss(targets, outputs):
                mu = outputs[...,0:1]
                sigma = soft(outputs[...,1:2])
                tau = soft(outputs[...,2:3])
                nu = soft(outputs[...,3:4])
                y = targets[...,0:1]
                mu_hat = targets[...,1:2]
                sigma_hat = targets[...,2:3]
                p1 = - K.log(sigma) - 0.5 * K.square((y - mu) / sigma)
                p2 = - K.log(tau) - 0.5 * K.square((mu_hat - mu) / tau)
                p3 = nu / 2 * K.log(2 * K.square(sigma) / nu) + \
                    2 * (nu / 2 -  1) * K.log(sigma_hat) - \
                    (K.square(sigma_hat) * nu) / (2 * K.square(sigma)) - \
                    tf.math.lgamma(nu / 2)
                    
                return K.sum(- p1 - p2 - p3 )
            
            def total_loss(targets, outputs):
                return loss_1(targets,outputs) + loss_2(targets, outputs) \
                       + loss_3(targets, outputs)
            
            c_2 = 1 / len(X_train_2)
            inputs = Input(shape=(1))
            inter = Dense(n_hidden_2[0], activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(c_2))(inputs)
            for i in range(len(n_hidden_2) - 1):
                inter = Dense(n_hidden_2[i+1], activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(c_2))(inter)
            outputs = Dense(4, kernel_regularizer=keras.regularizers.l2(c_2), 
                            activation = 'linear')(inter)
            model_2 = Model(inputs, outputs)
            
            model_2.compile(loss = explicit_loss, optimizer='adam')
            model_2.fit(X_train_2, targets, batch_size = len(X_train_2),
                        epochs = n_epochs_2, verbose = verbose2)
            self.model_2 = model_2
       
#%%
X_train, Y_train, X_test, Y_test = get_data(10000, 200)            
n_hidden = np.array([50, 50])
models = Neural_network(X_train, Y_train, n_hidden,
                        n_hidden_2 = np.array([50, 50]),n_epochs = 300,
                        n_epochs_2 = 300, split = 0.2)

model = models.model
uncertainty_model = models.model_2


x = np.array([-0.5])
print(" predicted y =", model.predict(x)[:,0], 
      "\n real y =", y(x), 
      "\n predicted tau =", soft(uncertainty_model.predict(x)[0][2:3]),
      "\n predicted sigma =", soft(model.predict(x)[:,1]),
      "\n real sigma =", sigma(x),
      "\n predicted nu =", soft(uncertainty_model.predict(x)[0][3:]))

n_hidden = np.array([50])
N_tests = 50
x_tests = np.linspace(-1, 1, 5)
results = np.zeros((N_tests, 5))
for i in range(N_tests):
    X_train, Y_train, X_test, Y_test = get_data(10000, 200)          

    models = Neural_network(X_train, Y_train, n_hidden, 
                            n_hidden_2 = np.array([100]), n_epochs = 300,
                            n_epochs_2 = 300, split = 0.5, verbose = False)
    model = models.model
    uncertainty_model = models.model_2
    for j, x in enumerate(x_tests):
        results[i, j] = model.predict(np.array([x]))[:,0] - y(x) / (
                soft(uncertainty_model.predict(np.array([x]))[:,2])).numpy()
    print(i, "/", N_tests)    
plt.hist(results[:,1])
np.std(results, axis = 0)
        

plt.plot(X_test, model.predict(X_test)[:,0], label = 'predicted')
plt.plot(X_test, y(X_test), label = 'true')
plt.legend()
plt.title('predicted y ')
plt.show()

plt.plot(X_test, soft(uncertainty_model.predict(X_test)[:,2]).numpy(),
         label = 'predicted')
#plt.plot(X_test, y(X_test), label = 'true')
plt.legend()
plt.title('predicted y ')
plt.show()
