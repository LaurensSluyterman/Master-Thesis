#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:20:41 2020

@author: Laurens Sluijterman
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



class Neural_network: 
    
    """
   This class represents a model.
   
   Parameters:
       X_train: A matrix containing the features of the training data.
       Y_train: A matrix containing the targets of the training data
       n_hidden: The number of hidden units in the hiddne layer
       n_epochs: The number of training epochs
   
    """
    
    def __init__(self, X_train, Y_train, n_hidden, n_epochs = 20,
                 uncertainty_estimates = True, n_hidden_2 = 0,
                 n_epochs_2 = 0, verbose = True):
        if n_hidden_2 == 0:
            n_hidden_2 = n_hidden
        if n_epochs_2 == 0:
            n_epochs_2 = n_epochs
        if uncertainty_estimates == True:
            X_train, X_train_2, Y_train, Y_train_2 = train_test_split(
                    X_train, Y_train, test_size=0.2)
           
        def Loss(y, musigma):
            dist = tfd.Normal(loc = musigma[..., :1], scale = 1e-3 +\
                              tf.math.softplus(musigma[...,1:]))
            return -np.sum(dist.log_prob(y))
        
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
        
        model.compile(loss = Loss, optimizer='adam')
        model.fit(X_train, Y_train, batch_size = 100, epochs = n_epochs, 
                  verbose = verbose)
        self.model = model      
 
        def get_uncertainty(self, X_train_2,Y_train_2, n_hidden_2, n_epochs_2):
            mu_hat = model_1.predict(X_train_2)[:,0]
            sigma_hat_squared =  (np.log(1
                             + np.exp(model_1.predict(X_train_2)[:,1])) \
                             + 1e-3) ** 2
            targets = np.stack((Y_train_2, mu_hat, sigma_hat_squared), axis = 1)
            def loss_1(targets, outputs):
                dist_1 = tfd.Normal(loc = outputs[..., :1], scale = 1e-5 +\
                                  tf.math.softplus(outputs[...,1:2]))
                return -np.sum(dist_1.log_prob(targets[...,:1]))
            def loss_2(targets, outputs):
                dist_2 = tfd.Normal(loc = outputs[..., 1:2], scale = 1e-5 +\
                                  tf.math.softplus(outputs[...,2:3]))
                return -np.sum(dist_2.log_prob(targets[...,1:2]))
            def loss_3(targets, outputs):
                sigma = 1e-5 + tf.math.softplus(outputs[...,1:2])
                v = 1e-5 + tf.math.softplus(outputs[...,3:4])
                concentration = v /2
                rate = 1 / (2 * sigma ** 2 / v)
                dist_3 = tfd.Gamma(concentration = concentration, 
                                   rate = rate)
                return -np.sum(dist_3.log_prob(targets[..., 2:]))
            def total_loss(targets, outputs):
                return loss_1(targets,outputs) + loss_2(targets, outputs) \
                       + loss_3(targets, outputs)
            c = 0
            inputs = Input(shape=(1))
            inter = Dense(n_hidden_2[0], activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(c))(inputs)
            for i in range(len(n_hidden_2) - 1):
                inter = Dense(n_hidden_2[i+1], activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(c))(inter)
            outputs = Dense(4, kernel_regularizer=keras.regularizers.l2(c), 
                            activation = 'linear')(inter)
            model_2 = Model(inputs, outputs)
            
            model_2.compile(loss = total_loss, optimizer='adam')
            model_2.fit(X_train_2, targets, batch_size = 100,
                        epochs = 10, verbose = True)
        if uncertainty_estimates == True:
            self.model_2 = get_uncertainty(self, X_train_2, Y_train_2,
                                           n_hidden_2, n_epochs_2)
#%%
n_hidden = np.array([50])
n_hidden_2 = n_hidden
modelc = Neural_network(X_train, Y_train, n_hidden, n_epochs = 30)
test_model = modelc.model_2
X_train_2, Y_train_2, d, e = get_data(2000, 0)
model_1 = model
mu_hat = model_1.predict(X_train_2)[:,0]
sigma_hat_squared =  (np.log(1
                 + np.exp(model_1.predict(X_train_2)[:,1])) \
                 + 1e-3) ** 2
targets = np.stack((Y_train_2, mu_hat, sigma_hat_squared), axis = 1)
def loss_1(targets, outputs):
    dist_1 = tfd.Normal(loc = outputs[..., :1], scale = 1e-5 +\
                      tf.math.softplus(outputs[...,1:2]))
    return -np.sum(dist_1.log_prob(targets[...,:1]))
def loss_2(targets, outputs):
    dist_2 = tfd.Normal(loc = outputs[..., 1:2], scale = 1e-5 +\
                      tf.math.softplus(outputs[...,2:3]))
    return -np.sum(dist_2.log_prob(targets[...,1:2]))
def loss_3(targets, outputs):
    sigma = 1e-5 + tf.math.softplus(outputs[...,1:2])
    v = 1e-5 + tf.math.softplus(outputs[...,3:4])
    concentration = v /2
    rate = 1 / (2 * sigma ** 2 / v)
    dist_3 = tfd.Gamma(concentration = concentration, 
                       rate = rate)
    return -np.sum(dist_3.log_prob(targets[..., 2:]))
def total_loss(targets, outputs):
    return loss_1(targets,outputs) + loss_2(targets, outputs) \
           + loss_3(targets, outputs)
c = 0
inputs = Input(shape=(1))
inter = Dense(n_hidden_2[0], activation='relu', 
              kernel_regularizer=keras.regularizers.l2(c))(inputs)
for i in range(len(n_hidden_2) - 1):
    inter = Dense(n_hidden_2[i+1], activation='relu', 
                  kernel_regularizer=keras.regularizers.l2(c))(inter)
outputs = Dense(4, kernel_regularizer=keras.regularizers.l2(c), 
                activation = 'linear')(inter)
model_2 = Model(inputs, outputs)

model_2.compile(loss = total_loss, optimizer='adam')
model_2.fit(X_train_2, targets, batch_size = 100,
            epochs = 10, verbose = True)