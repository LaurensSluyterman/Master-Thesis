#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:15:40 2020

@author: laurens
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
import time
import tensorflow.keras.backend as K

#%% Functions
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
   
   TODO: Look up how to correctly add docstring to this class
    """
    def __init__(self, X_train, y_train, n_hidden, n_epochs = 20, verbose = True):
       
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
        model.fit(X_train, y_train, batch_size = 100, epochs = n_epochs, 
                  verbose = verbose)
        self.model = model      
 

def density(model, X_train, Y_train):
    return np.exp(-model.evaluate(X_train, Y_train, verbose = 0))


def get_hypercube(model, h):
    
    """

    """
    
    weight_matrix_highs = []
    bias_highs = []
    weight_matrix_lows = []
    bias_lows = []
    for i in range(1,np.shape(model.layers)[0]): 
        weight_matrix = model.layers[i].get_weights()[0]
        bias_vector = model.layers[i].get_weights()[1]
        for j in range(np.shape(weight_matrix)[1]): 
            u = np.random.uniform()
            bias_highs.append(bias_vector[j] + h)
            bias_lows.append(bias_vector[j] - h * u)
            for k in range(np.shape(weight_matrix)[0]): 
                u = np.random.uniform()
                weight_matrix_highs.append(weight_matrix[k, j] + h)
                weight_matrix_lows.append(weight_matrix[k, j] - h * u)
    return [weight_matrix_lows, weight_matrix_highs, bias_lows, bias_highs]


def weight_to_list(weight):
    
    """
    TODO: FIX BIAS, something goes wrong
    """
    original_weights = model.get_weights()
    model.set_weights(weight)
    weights = []
    bias = []
    for i in range(1,np.shape(model.layers)[0]): 
        weight_matrix = model.layers[i].get_weights()[0]
        bias_vector = model.layers[i].get_weights()[1]
        for j in range(np.shape(weight_matrix)[1]): 
            bias.append(bias_vector[j])
            for k in range(np.shape(weight_matrix)[0]): 
                weights.append(weight_matrix[k, j])
    model.set_weights(original_weights)
    return weights, bias



def get_new_weight(hypercube):
    n_hidden = 50
    new_weight = np.array([\
              np.random.uniform(size = (1,n_hidden), 
                                low = hypercube[0][0: n_hidden], 
                                high = hypercube[1][0: n_hidden]),
              np.random.uniform(size = (n_hidden,), \
                                low = hypercube[2][0: n_hidden], \
                                high = hypercube[3][0: n_hidden]), \
              np.random.uniform(size = (n_hidden, 2), \
                                low =  np.reshape(hypercube[0][n_hidden:], 
                                                  (2, n_hidden)).T, \
                                high = np.reshape(hypercube[1][n_hidden:], (2, n_hidden)).T), \
              np.random.uniform(size = (2,), low = hypercube[2][n_hidden:],\
                                high = hypercube[3][n_hidden:])]) 
    return new_weight


def shrink_hypercube(hypercube, current_weight, previous_weight):
    hypercube2 = hypercube
    weights_old, bias_old = weight_to_list(previous_weight)
    weights_new, bias_new = weight_to_list(current_weight)
    for i in range(len(hypercube[0])):
        if weights_new[i] < weights_old[i]:
            hypercube2[0][i] = weights_new[i]
        else:
            hypercube2[1][i] = weights_new[i]
    for j in range(len(hypercube[2])):
        if bias_new[j] < bias_old[j]:
            hypercube2[2][j] = bias_new[j]
        else:
            hypercube2[3][j] = bias_new[j]
    return hypercube2        


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    " This code was written by mpariente and obtained from https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras"

    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(model._standardize_user_data(inputs, outputs))
    return output_grad

def shrink_hypercube_2(hypercube, model, current_weight, previous_weight,
                        threshold):
    hypercube2 = hypercube
    weights_old, bias_old = weight_to_list(previous_weight)
    weights_new, bias_new = weight_to_list(current_weight)
    gradients = get_weight_grad(model, X_train, Y_train)
    weight_gradient, bias_gradient = weight_to_list(gradients)
    for i in range(len(hypercube[0])):
        if (hypercube2[1][i] - hypercube[0][i]) * np.abs(weight_gradient[i]) > threshold:
            print('check1')
            if weights_new[i] < weights_old[i]:
                hypercube2[0][i] = weights_new[i]
            else:
                hypercube2[1][i] = weights_new[i]
    for i in range(len(hypercube[2])):
        if (hypercube2[3][i] - hypercube[2][i]) * np.abs(bias_gradient[i]) > threshold:
            print('check')
            if weights_new[i] < weights_old[i]:
                hypercube2[2][i] = bias_new[i]
            else:
                hypercube2[3][i] = bias_new[i]
            
    
    
def slice_sampler(N_samples, model, h, X_train, Y_train, gradient = False,
                  threshold = 0.01):
    if gradient == True:
        grads = model.optimizer.get_gradients(model.total_loss, 
                                              model.trainable_weights)
        symb_inputs = model._feed_inputs + model._feed_targets \
                        + model._feed_sample_weights
        f = K.function(symb_inputs, grads)
    original_weights = model.get_weights()
    weight_array = original_weights
    for i in range(0, N_samples):
        p_0 = density(model, X_train, Y_train)
        p_new = 0 
        u = np.random.uniform(low = 0, high = p_0)
        print(u)
        hypercube = get_hypercube(model, h)
        while p_new < u:          
            proposed_weight = get_new_weight(hypercube)
            model.set_weights(proposed_weight)
            p_new = density(model, X_train, Y_train)
            if p_new < u:
                if gradient == False:
                    hypercube = shrink_hypercube(hypercube, proposed_weight, 
                                                 weight_array[i])
                else:
                    model.set_weights(weight_array[i])
                    hypercube == shrink_hypercube_2(hypercube, model, 
                                                   proposed_weight,
                                                   weight_array[i],
                                                   threshold)
            else:
                weight_array = np.vstack((weight_array, proposed_weight))
                model.set_weights(proposed_weight)
        if (i / N_samples * 100 % 10) == 0:
            print('progress =', i / N_samples * 100 , '%', '\r', end='')
    model.set_weights(original_weights)

    return weight_array

def prediction_sampler_2(x, model, weight_array):
    
    """
    Add Docstring
    """
    
    original_weights = model.get_weights()
    mu_predictions = np.zeros(len(weight_array))
    sigma_predictions = np.zeros(len(weight_array))
    for i in range(len(weight_array)):
        model.set_weights(weight_array[i])
        mu_predictions[i]= model.predict(np.array([x]))[:,0]
        sigma_predictions[i] = np.log(1
                         + np.exp(model.predict(np.array([x]))[:,1][0])) \
                         + 1e-3 
    model.set_weights(original_weights)
    return mu_predictions, sigma_predictions
    

#%% Testing
    
X_train, Y_train, X_test, Y_test = get_data(10000, 200)

n_hidden = np.array([50])
modelc = Neural_network(X_train, Y_train, n_hidden, n_epochs = 30)
model = modelc.model

weight_array = slice_sampler(100, model, 0.1, X_train, Y_train, 
                             gradient = True, threshold = 0.001) #6 / s
x = 0.9
mu_test, sigma_test = prediction_sampler_2(x, model, weight_array)
print("x =", x,
      "prediction =", model.predict(np.array([x]))[:,0],
      "real = ", y(x),
      "\n predicted stdev at x = ", np.std(mu_test))

#Er gaat ergens iets mis met de weights, de predictie op het eind is heel raar..