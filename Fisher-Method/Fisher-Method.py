#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:08:49 2020

@author: laurens
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
#%% Making a class that contains the model

class net2: 
    def __init__(self, X_train, y_train, n_hidden, n_epochs = 2):
       
        def Loss(y, musigma):
            dist = tfd.Normal(loc = musigma[..., :1], scale = 1e-3 +\
                              tf.math.softplus(0.05 * musigma[...,1:]))
            return -dist.log_prob(y) 
        
        inputs = Input(shape=(1))
        inter = Dense(n_hidden, activation='elu', kernel_regularizer=keras.regularizers.l2(0.0))(inputs)
        outputs = Dense(2, kernel_regularizer=keras.regularizers.l2(0.0))(inter)

        model = Model(inputs, outputs)
        model.compile(loss = Loss, optimizer='sgd')
        model.fit(X_train, y_train, batch_size = 50, epochs = n_epochs)
        self.model = model
        
        self.weights = self.model.trainable_weights # weight tensors
       # weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
        self.gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.weights) # gradient tensors



#%%
n_hidden = 50
modelc = net2(Xtrain, Ytrain, n_hidden, n_epochs = 20)
model = modelc.model

    
grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
symb_inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
f = K.function(symb_inputs, grads)

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    " This code was written by mpariente and obtained from https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras"

    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(model._standardize_user_data(inputs, outputs))
    return output_grad

Fisher = np.zeros(shape = (4 * n_hidden + 2, 4 * n_hidden + 2), dtype = 'float64')
for i in range(len(Xtest)):
    grads = get_weight_grad(model, np.array([Xtest[i]]), np.array([Ytest[i]]))
    W1 = grads[0].reshape(n_hidden,1) #Weight matrix 
    b = grads[1].reshape(n_hidden,1) #bias matrix
    W2 = grads[2].reshape(2 * n_hidden,1) #Weightmatrix 2
    b2 = grads[3].reshape(2, 1)
    gradstotal = np.concatenate((W1, b, W2, b2))
    gradstotalT = np.transpose(gradstotal)
    Fisher += gradstotal.dot(gradstotalT)
Fisher = Fisher / len(Xtest)
Vars = np.diag(np.linalg.inv(Fisher))
print(Vars)
#%% Adding Noise to the parameters

np.diag(Fisher)
plt.plot(Xtest, model.predict(Xtest)[:,0])

a = np.array(model.get_weights())       # save weights in a np.array of np.arrays
b = np.array([np.random.normal(size = (1,n_hidden)),\
          np.random.normal(size = (n_hidden,)),\
          np.random.normal(size = (n_hidden, 2)),\
          np.random.normal(size = (2,))]) 
c = a + b # Add noise (note that I should still add the correct scales but here I am stuck)
model.set_weights(c) # Set the new weights