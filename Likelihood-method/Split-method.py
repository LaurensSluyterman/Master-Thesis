"""
Created on Thu Apr 16 14:20:41 2020.

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
import tensorflow.keras.backend as K
l2 = keras.regularizers.l2
import scipy
#%%
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
    """Return log(1 + exp(x)) + 1e-6."""
    return 1e-6 + tf.math.softplus(x)

class Neural_network: 
    """
    This class represents a model with extra networks to give uncertainties.
    
    Insiert longer description here
   
    Attributes:
        model: The trained network on all of the data, it outputs mu_hat
                and sigma_hat
        model_tau_i: A network trained to predict the uncertainty in mu_hat
        model_nu_i: A network trained to predict the uncertainty in sigma_hat
    
    
    Methods:
       mu_uncertainty: Gives the predicted uncertainty of mu_hat at a given
                value of x.
       sigma_uncertainty: Gives the predicted uncertainty of sigma_hat at a
                given value of x. 
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
                 n_epochs_2 = 20, verbose = True):
        """ 
        Parameters
            X_train: A matrix containing the inputs of the training data.
            Y_train: A matrix containing the targets of the training data.
            n_hidden (array): An array containing the number of hidden units
                     for each hidden layer. The length of this array 
                     specifies the number of hidden layers used for the 
                     training of the main model.
            n_hidden_2 (array): An array containing the number of hidden units
                     for each hidden layer used in the models that predict
                     the uncertainties. 
            n_epochs_i: The amount of epochs used in training.
            verbose (boolean): A boolean that determines if the training-
                    information is displayed.
        """   
        X_train_1, X_train_2, Y_train_1, Y_train_2 = train_test_split(
                    X_train, Y_train, test_size = 2 / 3)
        X_train_2, X_train_3, Y_train_2, Y_train_3 = train_test_split(
                    X_train_2, Y_train_2, test_size = 0.5)
        
        def loss(targets, outputs):
            """
            Give the loss function used for the main model.
            
            This function calculates the negative loglikelihood of a
            normal distribution with mean mu_hat and stdev sigma_hat
            """
            mu = outputs[...,0:1]
            sigma = soft(outputs[...,1:2])
            y = targets[...,0:1]
            l = - K.log(sigma) - 0.5 * K.square((y - mu) / sigma)
            return - l
        
        def loss_tau(targets, outputs):
            """
            Give the loss function used for the tau-models.
            
            This function calculates the negative loglikelihood of a 
            normal distribution with mean mu_hat and variance tau.
            """
            tau = soft(outputs)
            l = -K.log(tau) - 0.5 * K.square((targets[...,:1] - 
                                              targets[...,1:])  / tau)
            return  - l
            
        def loss_nu(targets, outputs):
            """
            Give the loss function used for the nu-models.
            
            This function calculates the negative loglikelihood of 
            a gamma(shape = nu / 2, scale = 2 / nu) distribution.
            """
            nu = 1 + soft(outputs)
            sigma = targets[...,:1]
            sigma_hat = targets[...,1:]
            x = (sigma_hat / sigma) ** 2
            l = - tf.math.lgamma(nu / 2) - (nu / 2) * K.log(2 / nu) + \
                (nu / 2 - 1) * K.log(x) - x / 2 * nu
            return - l
        
        def get_model(n_hidden, c, loss, n_out):
            """
            Construct an untrained neural network.
            
            This function constructs and compiles an untrained neural network
            with a specific shape, loss, and regularization.
                
            Parameters:
                n_hidden (array): Array containing the amount of neurons in 
                    each hidden unit.
                c: The factor used for l2-regularization in each hidden layer.
                loss: The loss function that is used. 
                n_out: The number of parameters that the network outputs. 
                
            Returns:
                model: An (untrained) model.
                
            """
            inputs = Input(shape=(1))
            inter = Dense(n_hidden[0], activation='elu', 
                      kernel_regularizer = l2(c),
                      bias_regularizer = l2(0))(inputs)
            for i in range(len(n_hidden) - 1):
                inter = Dense(n_hidden[i+1], activation='elu', 
                              kernel_regularizer=keras.regularizers.l2(c),
                              bias_regularizer = l2(0))(inter)
            outputs = Dense(n_out, activation = 'linear')(inter)
            model = Model(inputs, outputs) 
            model.compile(loss = loss, optimizer='adam')
            return model
        
        def get_targets(model_a, model_b, X_a, X_b):
            mu_own = np.vstack((model_a.predict(X_a), 
                                model_b.predict(X_b)))[:,0]
            mu_other = np.vstack((model_b.predict(X_a), 
                                model_a.predict(X_b)))[:,0]
            sigma_own = np.vstack((soft(model_a.predict(X_a)),
                                  soft(model_b.predict(X_b))))[:,1]
            sigma_other = np.vstack((soft(model_b.predict(X_a)),
                                  soft(model_a.predict(X_b))))[:,1]
            tau_targets = np.stack((mu_own, mu_other), axis = 1)
            nu_targets = np.stack((sigma_own, sigma_other), axis = 1)
            return tau_targets, nu_targets

        model = get_model(n_hidden, 0, loss, 2)
        model_1 = get_model(n_hidden, 0, loss, 2)
        model_2 = get_model(n_hidden, 0, loss, 2)
        model_3 = get_model(n_hidden, 0, loss, 2)
        model_tau_1 = get_model(n_hidden, 0, loss_tau, 1)
        model_tau_2 = get_model(n_hidden, 0, loss_tau, 1)
        model_tau_3 = get_model(n_hidden, 0, loss_tau, 1)
        model_nu_1 = get_model(n_hidden, 0, loss_nu, 1)
        model_nu_2 = get_model(n_hidden, 0, loss_nu, 1)
        model_nu_3 = get_model(n_hidden, 0, loss_nu, 1)
        
        model.fit(X_train, Y_train, batch_size = 100, epochs = n_epochs, 
                  verbose = verbose)
        model_1.fit(X_train_1, Y_train_1, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        model_2.fit(X_train_2, Y_train_2, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        model_3.fit(X_train_3, Y_train_3, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        
            
        tau_12_targets, nu_12_targets = get_targets(model_1, model_2, 
                                                    X_train_1, X_train_2)
        tau_13_targets, nu_13_targets = get_targets(model_1, model_3,
                                                    X_train_1, X_train_3)
        tau_23_targets, nu_23_targets = get_targets(model_2, model_3, 
                                                    X_train_2, X_train_3)
        
        model_tau_1.fit(np.hstack((X_train_1, X_train_2)), 
                          tau_12_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_tau_2.fit(np.hstack((X_train_1, X_train_3)), 
                          tau_13_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_tau_3.fit(np.hstack((X_train_2, X_train_3)), 
                          tau_23_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_1.fit(np.hstack((X_train_1, X_train_2)), 
                          nu_12_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_2.fit(np.hstack((X_train_1, X_train_3)), 
                          nu_13_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_3.fit(np.hstack((X_train_2, X_train_3)), 
                          nu_23_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        
        self.model = model   
        self.model_tau_1 = model_tau_1
        self.model_tau_2 = model_tau_2
        self.model_tau_3 = model_tau_3
        self.model_nu_1 = model_nu_1
        self.model_nu_2 = model_nu_2
        self.model_nu_3 = model_nu_3
            
    def mu_uncertainty(self, x): 
        """
        Give the predicted uncertainty of mu_hat at x.
        
        This function calculates the average of the predictions of the 
        three tau_models that were previously trained. 
        """
        tau = soft(self.model_tau_1.predict(x)) + soft(self.model_tau_2.predict(x)) + \
                soft(self.model_tau_3.predict(x))
        return tau.numpy() / 3
        
    def sigma_uncertainty(self, x):
        """
        Give the predicted uncertainty of sigma_hat at x.
        
        This function calculates the average of the predictions of the 
        three nu_models that were previously trained. 
        """
        nu = soft(self.model_nu_1.predict(x)) + soft(self.model_nu_2.predict(x)) + \
                soft(self.model_nu_3.predict(x))
        return nu.numpy() / 3 + 1

def confidence_bound(sigma, nu):
    """Create a conficence interval for a given sigma and nu."""
    low = sigma *  np.sqrt(scipy.stats.chi2.ppf(q = 0.17, df = nu) / nu)
    high = sigma * np.sqrt(scipy.stats.chi2.ppf(q = 0.83, df = nu) / nu)
    return [low, high]            
       
#%% Testing
X_train, Y_train, X_test, Y_test = get_data(20000, 200)            
models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 20]),
                        n_hidden_2 = np.array([50, 50, 20]),n_epochs = 20,
                        n_epochs_2 = 20)

model = models.model

y(0.3)
x = np.array([-1])
print(" predicted y =", model.predict(x)[:,0], 
      "\n real y =", y(x), 
      "\n predicted tau =", models.mu_uncertainty(x)[0],
      "\n predicted sigma =", soft(model.predict(x)[:,1]),
      "\n real sigma =", sigma(x),
      "\n confidence interval =", confidence_bound(soft(model.predict(x)[:,1]).numpy(),
                                    models.sigma_uncertainty(x)[0]))


N_tests = 30
N = 40
x_tests = np.linspace(-1, 1, N)
results = np.zeros((N_tests, N))

for i in range(N_tests):
    
    X_train, Y_train, X_test, Y_test = get_data(20000, 0)          

    models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 30]), 
                            n_hidden_2 = np.array([50, 50, 30]), n_epochs = 30,
                            n_epochs_2 = 30, verbose = False)
    model = models.model
    for j, x in enumerate(x_tests):
        results[i, j] = model.predict(np.array([x]))[:,0] - y(x) / (
                soft(models.mu_uncertainty(np.array([x]))[0])).numpy()
    print(i, "/", N_tests)    
plt.hist(results[:,24])
np.std(results, axis = 0)
        



#%%