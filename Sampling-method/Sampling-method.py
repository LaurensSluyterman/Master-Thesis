#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:46:50 2020

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
import time

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
       
        l2 = keras.regularizers.l2
        def Loss(y, musigma):
            dist = tfd.Normal(loc = musigma[..., :1], scale = 1e-3 +\
                              tf.math.softplus(musigma[...,1:]))
            return -np.sum(dist.log_prob(y))
        
        c = 1 / (len(X_train))
        inputs = Input(shape=(1))
        inter = Dense(n_hidden[0], activation='relu', 
                      kernel_regularizer = l2(c), 
                      bias_regularizer = l2(c))(inputs)
        for i in range(len(n_hidden) - 1):
            inter = Dense(n_hidden[i+1], activation='relu', 
                          kernel_regularizer = l2(c), 
                          bias_regularizer = l2(c))(inter)
        outputs = Dense(2, kernel_regularizer = l2(c), 
                        bias_regularizer = l2(c),
                        activation = 'linear')(inter)
        model = Model(inputs, outputs)
        
        model.compile(loss = Loss, optimizer='adam')
        model.fit(X_train, y_train, batch_size = 200, epochs = n_epochs, 
                  verbose = verbose)
        self.model = model      
 



def get_scales(model, h_guess, difference, X_train, Y_train):
    
    """
    This function calculates how much an individual weight has to be changed
    in order to make the likelihood on the test set be a factor 'difference'
    larger. If the model finds that the weight is not in a minimum, 
    we will give it a larger scale. 
    
    parameters:
        model: A trained tensorflow model
        h_guess (float): An initial guess for the change
        difference (float): The desired difference in log likelihood
        X_train, Y_train: Arrays containing the x and y values of the test set
        
    returns:
        weight_matrix_scales: A list containing all the scales for the 
        weights
        bias_scales: A list containing all the scales for the weights.
        
    TODO: 
        Think if I should use the test or training set for the scales?
        ADD optimizer! Too many weights are zero if the model is worthless.
    """
    
    weight_matrix_scales = []
    bias_scales = []
    original_weights = model.get_weights()
    L_1 = model.evaluate(X_train, Y_train, verbose = 0) \
          - sum(model.losses).numpy()
    for i in range(1,np.shape(model.layers)[0]): 
        weight_matrix = model.layers[i].get_weights()[0]
        bias_vector = model.layers[i].get_weights()[1]
        for j in range(np.shape(weight_matrix)[1]): 
         #   bias_vector[j] += h_guess * bias_vector[j]
            bias_vector[j] += h_guess
            model.layers[i].set_weights([weight_matrix, bias_vector])
            L_2 = model.evaluate(X_train, Y_train, verbose = 0) \
                  - sum(model.losses).numpy()
            #bias_vector[j] =  bias_vector[j] / (1 + h_guess)
            bias_vector[j] -= h_guess
            model.set_weights(original_weights)
            c = (L_2 - L_1) / ((h_guess ) ** 2)
            if c <= 0:
                bias_scales.append((L_1 - L_2) / (h_guess)) 
                print('H too small')
            else:
                bias_scales.append(np.sqrt(np.log(difference) \
                                        / np.abs(c)))
            for k in range(np.shape(weight_matrix)[0]): 
               # weight_matrix[k, j] += (1 + h_guess) * weight_matrix[k, j]
                weight_matrix[k, j] += h_guess
                model.layers[i].set_weights([weight_matrix, bias_vector])
                L_2 = model.evaluate(X_train, Y_train, verbose = 0) \
                      - sum(model.losses).numpy()
               # weight_matrix[k, j] = weight_matrix[k, j] / (1 + h_guess)
                weight_matrix[k, j] -=  h_guess
                model.set_weights(original_weights)
                c = ((L_2 - L_1) / ((h_guess * weight_matrix[k, j]) ** 2 ))
                if c <= 0:
                    print('H too small')
                    weight_matrix_scales.append((L_1 - L_2) / (h_guess)) 
                else:
                    weight_matrix_scales.append(np.sqrt(np.log(difference)\
                                                    / (np.abs(c))))
    return weight_matrix_scales, bias_scales
            

def get_perturbation(original_weights, weight_scales, bias_scales):
    
    """
    This function calculates a multidemensional noise term where each
    component has a specific scale calculated with 'get_scales'.
    
    parameters:
        original_weights: The original (trained) weights of the model
        weight_scales, bias_scales: The scales obtained by 'get_scales'
        
    
    returns:
        perturbations: A noise sample from U[-scales, scales]       
    """
    n_hidden = 50
    perturbation = np.array([\
              np.random.uniform(size = (1,n_hidden), \
                                low = [i * -1 for i in weight_scales[0:n_hidden]], \
                                high = weight_scales[0: n_hidden]),\
              np.random.uniform(size = (n_hidden,), \
                                low = [i * -1 for i in bias_scales[0:n_hidden]], \
                                high = bias_scales[0:n_hidden]), \
              np.random.uniform(size = (n_hidden, 2), \
                                low =  -1 * np.reshape(weight_scales[n_hidden:], (2, n_hidden)).T, \
                                high = np.reshape(weight_scales[n_hidden:], (2, n_hidden)).T), \
              np.random.uniform(size = (2,), low = [i * -1 for i in bias_scales[n_hidden:]],\
                                high = bias_scales[n_hidden:])]) 
    return perturbation


def get_densities(N_samples, model, X_train, Y_train, weight_scales, bias_scales):
    
    """
    This function outputs a set of weights with corresponding densities. It
    does this by using the loss function which equals the density of the data 
    given the weights times the density of the weights. The function 
    normalizes the densities before returning them. The function only
    accepts weights that result in a likelihood that is at least 1 percent
    of the likelihood of the data given the trained weights. 
    
    Parameters:
        N_samples (int): The number of weights to be sampled
        model: The trained model whose weights are to be sampled
        X_train, Y_train: The inputs and targets used in training.
        weight_scales: A multidimensional array generated with 'get_perturbation'.
                      This determens the size of the noise that is added to
                      the weight matrices.
        bias_scales: A multi dimensional array generated witn 'get_perturbation'.
                     This determens the size of the noise that is added to
                     the bias vectors.
                     
    Returns:
        Normalized densities: A list containing the densities of the weights.
        weight_array: An array containing the sampled weights. 
    """
    
    original_weights = model.get_weights()
    weight_array = original_weights
    densities = [np.exp( - model.evaluate(X_train, Y_train, verbose = 0))]


    start = time.time()
    for i in range(N_samples - 1):
        perturbation =  get_perturbation(original_weights, weight_scales, 
                                        bias_scales)
        new_weight = original_weights + perturbation
        model.set_weights(new_weight)
        density = np.exp(- model.evaluate(X_train, Y_train, verbose = 0 ))
        while density < densities[0] / 10:
            perturbation = perturbation / 2
            new_weight = original_weights + perturbation
            model.set_weights(new_weight)
            density = np.exp(- model.evaluate(X_train, Y_train, verbose = 0 ))
        weight_array = np.vstack((weight_array,new_weight))
        densities.append(density)
    end = time.time()
    normalization_constant = np.sum(densities) 
    model.set_weights(original_weights)
  #  print(end - start) #N = 100 takes 1.6 s
    #N = 5000 took 703s
    return densities / normalization_constant, weight_array


def expectation(x, densities, weight_array, original_weights):
    
    """
    This function returns the expectation of (mu(x)_omega_hat -
    mu(x)_omega)^2 with respect to the distribution of the weights that was
    learned using 'get_densities'
    
    Parameters: 
        x (float): The x value at which the uncertainty estimate
                   is desired.
        densities: List of densities that was calculated using 'get_densities'.
        weight_array: Array of weights that was calculated using 
                      'get_densities'.
        original_weights: The original_weights of the trained model.
    
    Returns:
        expectation (float): the expectation of (mu(x)_omega_hat -
                             mu(x)_omega)^2.                     
    """
    
    model.set_weights(original_weights)
    mu_omega_hat = model.predict(np.array([x]))[:,0]
    expectation = 0
    for i in range(0, len(densities)):
        model.set_weights(weight_array[i])
        mu_omega = model.predict(np.array([x]))[:,0]
        expectation += ((mu_omega - mu_omega_hat) ** 2) * densities[i] 
    model.set_weights(original_weights)
    return expectation[0]

def expectation2(x, densities, weight_array, original_weights):
    
    """ 
    This is the same function as 'expectation but now the expectation of
    (sigma(x)_omega_hat - sigma(x)_omega)^2 is evaluated.   
    """
    model.set_weights(original_weights)
    sigma_omega_hat = np.log(1 + np.exp(model.predict(np.array([x]))[:,1][0])) \
                          + 1e-3
    expectation = 0
    for i in range(0, len(densities)):
        model.set_weights(weight_array[i])
        sigma_omega = np.log(1 + np.exp(model.predict(np.array([x]))[:,1][0])) \
                      + 1e-3
        expectation += (sigma_omega - sigma_omega_hat) ** 2 * densities[i]
    model.set_weights(original_weights)
    return expectation

def prediction_sampler(x, N, densities, weight_array, original_weights):
    
    """
    This function calculates the predictions corresponding to
    a weight array that consists of weights sampeld uniformely from (a 
    relevant part of) the paremeter space. 
    
    Parameters:
        x: The x value at which the predictions are evaluated
        model: The model that we are using
        weight_array: An array of weights that should be sampeled 
                      uniformly from the paremter space 
        densities: An array containing the densities of the weights in 
                   'weight_array'.
        original_weights: The original weights of the model after training
    """
    indices = np.arange(0, len(densities))
    draw = np.random.choice(indices, N, p=densities, replace = True)
    mu_predictions = np.zeros(N)
    sigma_predictions = np.zeros(N)
    for i, j in enumerate(draw):
        model.set_weights(weight_array[j])
        mu_predictions[i]= model.predict(np.array([x]))[:,0]
        sigma_predictions[i] = np.log(1
                         + np.exp(model.predict(np.array([x]))[:,1][0])) \
                         + 1e-3 
        
    model.set_weights(original_weights)
    return mu_predictions, sigma_predictions

def prediction_sampler_2(x, model, weight_array):    
    """
    This function calculates the predictions corresponding to
    a weight array that consists of weights sampeld from the posterior
    weight distribution. 
    
    Parameters:
        x: The x value at which the predictions are evaluated
        model: The model that we are using
        weight_array: An array of weights that should be sampeled 
                      from the posterior weight distribution. p(w|D)
    
    Returns:
        The predictions corresponding to the weights provided by weight_array
        at a given x value.
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

def proposal_distribution(weights, scale):
    
    """
    This function gives the proposal distribution that is used by
    'metropolis_hastings'.
    
    Parameters: 
        weights: The current weight.
        scale: The standard deviation of the normal noise that is added
               to the weights. 
    
    Returns: A perturbed version of the input weight
    """   
    n_hidden = 50 
    perturbation = np.array([\
              np.random.normal(size = (1, n_hidden), loc = 0, scale = scale)    ,\
              np.random.normal(size = (n_hidden), loc = 0, scale = scale)    , \
              np.random.normal(size = (n_hidden, 2), loc = 0, scale = scale)    , \
              np.random.normal(size = (2,), loc = 0, scale = scale)])    
    return weights + perturbation

def metropolis_hastings(model, N_samples, scale, X_train, Y_train):
    
    """
    This functions provides a basic inplementation of a metropolis hastings
    algeorithm. 
    
    Parameters:
        model: The model that is used.
        N_samples: The number of samples that are obtained from p(w|D). 
        scale: The standarddeviation of the normal distribution that is used
                as the approximate distribution q(w).
        X_train: Array containing the inputs.
        Y_train: Array containing the targets.
        
    Returns:
        weight_array: An array containing weights that are sampled from p(w|D)
        
    N.B. this implementation of MH does not work, the weights are not
    in dependent and the scale that is given has an enourmous influence 
    in the results. 
    """
    
    original_weights = model.get_weights()
    weight_array = original_weights
    weight_old = original_weights
    weight_scales, bias_scales = get_scales(model, 0.1, 1000, 
                                            X_train, Y_train)
    p_old = np.exp(-model.evaluate(X_train, Y_train, verbose = 0))
    accepted = 0
    for i in range(N_samples):
        perturbation =  get_perturbation(original_weights, weight_scales, 
                                        bias_scales)
        weight_proposed = original_weights + perturbation
        #weight_proposed = proposal_distribution(weight_old, scale)
        model.set_weights(weight_proposed)
        p_proposed = np.exp(-model.evaluate(X_train, Y_train, verbose = 0))
        alpha = p_proposed / p_old 
        print(alpha)
        if np.random.uniform(low = 0, high = 1) < np.min((1, alpha)):
            weight_array = np.vstack((weight_array,weight_proposed))
            weight_old = weight_proposed
            p_old = p_proposed
            accepted += 1
        else:
            weight_array = np.vstack((weight_array, weight_old))
      #      model.set_weights(weight_old)
    model.set_weights(original_weights)
    print("accepted percentage =", accepted / N_samples * 100)
    return(weight_array)
        


def test(x, N_simulations, N_samples, N_train, N_test, h_guess, difference):   
    """This function tests how well the sampling method works. 
    
    Parameters:
        x: The x value at which (prediction(x) - real(x)) / predicted stdev
           is evaluated.
        N_samples(int): The amount of simulations that is used.
        N_train(int): The amount of data-points per simulation.
        N_test(int): The amount of test points
        h_guess (int): The initial guess of the scale given to 'get_scales'.
        difference: The desired difference in likelihood given to 'get_scales'.
    """    
    testresults1_mu = np.zeros(N_simulations)
    testresults1_sigma = np.zeros(N_simulations)
        
    for i in range(0, N_simulations):
        Xtrain, Ytrain, Xtest, Ytest = get_data(N_train, 0)
        model = Neural_network(Xtrain, Ytrain, n_hidden = np.array([50]), 
                               verbose = False).model
        mu_hat = model.predict(np.array([x]))[:,0]
        sigma_hat = model.predict(np.array([x]))[:,1]                       
        original_weights = model.get_weights()
        weight_scales, bias_scales = get_scales(model, h_guess, difference, 
                                                X_train, Y_train)
        densities, weight_array = get_densities(N_samples, model, X_train, \
                                       Y_train, weight_scales, bias_scales)
        testresults1_mu[i] = (mu_hat - y(x)) / \
                    np.sqrt(expectation(x, densities,
                                        weight_array,original_weights))
        testresults1_sigma[i] = (sigma_hat - sigma(x)) / \
                    np.sqrt(expectation2(x, densities, 
                                        weight_array,original_weights))
        print(i / N_simulations * 100, '%', end='\r', flush=False)
    return testresults1_mu, testresults1_sigma\

def test2(x, N_simulations, N_samples, N_train, N_test, h_guess, difference):   
    """This function tests how well the sampling method works. 
    
    Parameters:
        x: The x value at which (prediction(x) - real(x)) / predicted stdev
           is evaluated.
        N_samples(int): The amount of simulations that is used.
        N_train(int): The amount of data-points per simulation.
        N_test(int): The amount of test points
        h_guess (int): The initial guess of the scale given to 'get_scales'.
        difference: The desired difference in likelihood given to 'get_scales'.
    """  
    testresults1_mu = np.zeros(N_simulations)
    testresults1_sigma = np.zeros(N_simulations)
    
   
    for i in range(0, N_simulations):
        Xtrain, Ytrain, Xtest, Ytest = get_data(N_train, 0)
        model = Neural_network(Xtrain, Ytrain, n_hidden = np.array([50]), 
                               verbose = False).model
        mu_hat = model.predict(np.array([x]))[:,0]
        sigma_hat = model.predict(np.array([x]))[:,1]                       
        original_weights = model.get_weights()
        weight_scales, bias_scales = get_scales(model, h_guess, difference, 
                                                X_train, Y_train)
        densities, weight_array = get_densities(N_samples, model, X_train, \
                                       Y_train, weight_scales, bias_scales)
        testmu, testsigma = prediction_sampler(x, 1000, densities, 
                                               weight_array, original_weights)
        testresults1_mu[i] = (mu_hat - y(x)) / np.std(testmu)
        testresults1_sigma[i] = (sigma_hat - sigma(x)) / np.std(testsigma)
        print(i / N_simulations * 100, '%', end='\r', flush=False)
    return testresults1_mu, testresults1_sigma

#%%% Testing
X_train, Y_train, X_test, Y_test = get_data(10000, 2000)
plt.plot(X_train,Y_train)
plt.plot(X_train,y(X_train))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

n_hidden = np.array([50])
modelc = Neural_network(X_train, Y_train, n_hidden, n_epochs = 50)
model = modelc.model

plt.plot(X_test, model.predict(X_test)[:,0], label = 'predicted')
plt.plot(X_test, y(X_test), label = 'true')
plt.legend()
plt.title('predicted y ')
plt.show()

plt.title('predicted sigma')
plt.plot(X_test, 1e-3 + np.log(1 + np.exp(model.predict(X_test)[:,1])), 
         label = 'predicted')
plt.plot(X_test, sigma(X_test), label = 'true')
plt.legend()
plt.show()
original_weights = np.array(model.get_weights())
model.set_weights(original_weights)    

#%% Testing Metropolis Hasting
weight_array = metropolis_hastings(model, 50, 10, X_train, Y_train)
x = .2
mu_test, sigma_test = prediction_sampler_2(x, model, weight_array)
print("x =", x,
      "prediction =", model.predict(np.array([x]))[:,0],
      "real = ", y(x),
      "\n predicted stdev at x = ", np.std(mu_test))




#%%            
weight_scales, bias_scales = get_scales(model, 0.1, 2, X_train, Y_train)
    
densities, weight_array = get_densities(100, model , X_train, \
                                       Y_train, weight_scales, bias_scales)
x = 0.6
print("x =", x,
      " prediction =",  model.predict(np.array([x]))[:,0][0], 
      "\n real value =", y(x) ,
      "\n predicted stdev at x =", np.sqrt(expectation(x, densities, weight_array,
                                        original_weights))
)


testmu, testsigma = prediction_sampler(x, 1000, densities, weight_array, original_weights)
print("x =", x,
      "prediction =", model.predict(np.array([x]))[:,0],
      "real = ", y(x),
      "\n predicted stdev at x = ", np.std(testmu))



x = -3
print("x =", x,
      " prediction =",  1e-3 + np.log(1 + np.exp(model.predict(np.array([x]))[:,1][0])), 
      "\n real value =", sigma(x) ,
      "\n predicted stdev at x =", np.sqrt(expectation2(x, densities, weight_array,
                                        original_weights)))

x = 0.6
mu_results, sigma_results = test(x, 50, 100, 10000, 0, 1 / 10000, 5000)


plt.title("x = {x}, sigma = {std}".format(x = x, std = round(np.std(mu_results),2)))
plt.hist(mu_results)
plt.show()

plt.title("x = {x}, sigma = {std}".format(x = x, std = round(np.std(sigma_results),2)))
plt.hist(sigma_results)
plt.show()
np.savetxt('test1mux=-1.out', mu_results, delimiter=',')
np.savetxt('test1sigmax=-1.out', sigma_results, delimiter=',')
