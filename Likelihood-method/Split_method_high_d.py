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

def soft(x):
    """Return log(1 + exp(x)) + 1e-6."""
    return 1e-6 + tf.math.softplus(x)

def y_difficult(x):
    "returns y"
    return 0.1 * x[0] + 0.3 * x[1]

def sigma_difficult(x):
    "returns sigma"
    return soft(0.1 * x[1] + 0.2 * x[2] - 0.05*  x[4] * x[5])
   # return 0.1 * np.exp(x1 - x2 + x3 * x4 + x5 ** 0.2) + 0.1

def get_fancy_data(N_train, N_test):
    """
    Get a difficult dataset.

    Parameters
    ----------
    N_train : TYPE
        DESCRIPTION.
    N_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    mean = [0, 0.1, 0.2, 0.3, 0.1, 0.5, -0.3]
    A = [[0.1,  0.4, 0, 0, 0,  0,  0], 
           [0.1,    0.3, 0, 0, 0,  0.1,   0],
           [0,    0, 0.2, 0,   0, 0, 0],
           [0.4,    0, 0,   0.2, 0, 0,  0.21],
           [0,    0, 0,   0,   0.4, 0, 0.2],
           [-0.2,    0, 0, -0.2, 0,     0.1, 0],
           [0,    0, 0.7, 0, 0, 0, 0.2]
           ]
    cov = np.transpose(A).dot(A)
    X_train = np.random.multivariate_normal(mean, cov, (N_train,))
    Y_train = np.zeros(N_train)
    X_test = np.random.multivariate_normal(mean, cov, (N_test,))
    Y_test = np.zeros(N_test)
    for i in range(N_train):
        Y_train[i] = y_difficult(X_train[i]) \
            + np.random.normal(0, sigma_difficult(X_train[i]))
    for i in range(N_test):
        Y_test[i] = y_difficult(X_test[i]) + \
                  np.random.normal(0, sigma_difficult(X_test[i]))
    return X_train, Y_train, X_test, Y_test 

    

class Neural_network: 
    """
    This class represents a model with extra networks to give uncertainties.
    
    In this class a total of 10 models are trained. The first model is 
    trained on the entire dataset and is used as our predictor. This model
    outputs mu_hat and sigma_hat and is trained using the loglikelihood of 
    the data assuming a normal distribution. 
    In order to get an estimate of the uncertainty of mu_hat and sigma_hat,
    9 more models are trained. First the data is split in three equal parts,
    then three more models are trained on these parts (with the same 
    structure as the main model). Then these 3 models are used to produce
    new targets hat(mu)_seen, hat(mu)_unseen, hat(sigma)_seen, 
    hat(sigma)_unseen for each pair that we can make by picking two models out
    of the three. This is done by letting the two model first predict on the
    part of the data they were trained on and then on the other part. 
    Using these new targets an additional three models are trained to obtain
    uncertainty estimates for mu_hat and sigma_hat.
   
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
                
    """
    
    def __init__(self, X_train, Y_train, n_hidden, n_epochs = 20,
                 uncertainty_estimates = True, n_hidden_2 = np.array([50]),
                 n_epochs_2 = 20, verbose = True):        
        """ 
        Parameters:
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
        input_shape = 7
        X_train_1, X_train_2, Y_train_1, Y_train_2 = train_test_split(
                    X_train, Y_train, test_size = 2 / 3)
        X_train_2, X_train_3, Y_train_2, Y_train_3 = train_test_split(
                    X_train_2, Y_train_2, test_size = 0.5)
        
        def loss(targets, outputs):
            """
            Give the loss function used for the main model.
            
            This function calculates the negative loglikelihood of a
            normal distribution with mean mu_hat and stdev sigma_hat.
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
            inputs = Input(shape=(input_shape))
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
            """
            Prepare the data for the uncertainty predicting networks.
            
            This function takes two models and makes the mu_seen and
            mu_unseen arrays. These are then used to predict tau and nu. 
            
            Parameters:
                model_a: The first model.
                model_b: The second model.
                X_a: The x-values that were used training model_a
                X_b: The x-values that were used training model_b       
            
            Returns:
                tau_targets: A (len(X_a) + len(X_b), 2) array contaning 
                    mu_seen and mu_unseen.
                nu_targets: A (len(X_a) + len(X_b), 2) array contaning 
                    sigma_seen and sigma_unseen.
                
            """
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

        model = get_model(n_hidden, 1 / len(X_train), loss, 2)
        model_1 = get_model(n_hidden, 1/ len(X_train_2), loss, 2)
        model_2 = get_model(n_hidden, 1 / len(X_train_2), loss, 2)
        model_3 = get_model(n_hidden, 1 / len(X_train_2), loss, 2)
        model_tau_1 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        model_tau_2 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        model_tau_3 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        model_nu_1 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_nu, 1)
        model_nu_2 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_nu, 1)
        model_nu_3 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_nu, 1)
        
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
        
        model_tau_1.fit(np.vstack((X_train_1, X_train_2)), 
                          tau_12_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_tau_2.fit(np.vstack((X_train_1, X_train_3)), 
                          tau_13_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_tau_3.fit(np.vstack((X_train_2, X_train_3)), 
                          tau_23_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_1.fit(np.vstack((X_train_1, X_train_2)), 
                          nu_12_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_2.fit(np.vstack((X_train_1, X_train_3)), 
                          nu_13_targets, batch_size = 100,
                          epochs = n_epochs_2, verbose = verbose)
        model_nu_3.fit(np.vstack((X_train_2, X_train_3)), 
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
        tau_1 = soft(self.model_tau_1.predict(x)).numpy()[0]
        tau_2 = soft(self.model_tau_2.predict(x)).numpy()[0]
        tau_3 = soft(self.model_tau_3.predict(x)).numpy()[0]
        return np.max([tau_1, tau_2, tau_3])
        
    def sigma_uncertainty(self, x):
        """
        Give the predicted uncertainty of sigma_hat at x.
        
        This function calculates the average of the predictions of the 
        three nu_models that were previously trained. 
        """
        nu_1 = soft(self.model_nu_1.predict(x)) + 1
        nu_2 = soft(self.model_nu_2.predict(x)) + 1
        nu_3 = soft(self.model_nu_3.predict(x)) + 1
        return np.median([nu_1, nu_2, nu_3])

def confidence_interval(sigma_hat, nu):
    """
    Create a conficence interval for a given sigma and nu.
    
    This function makes a confidence interval sigma given sigma_hat and
    nu_hat. 
    """
    high = sigma_hat  / np.sqrt(scipy.stats.chi2.ppf(q = 0.17, df = nu) / nu)
    low = sigma_hat / np.sqrt(scipy.stats.chi2.ppf(q = 0.83, df = nu) / nu)
    return [low[0], high[0]]            
       
#%% Testing
X_train, Y_train, X_test, Y_test = get_fancy_data(20000, 1000)                
models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 20]),
                        n_hidden_2 = np.array([50, 50, 20]),n_epochs = 50,
                        n_epochs_2 = 30)

model = models.model

i = 8
print(" predicted y =", model.predict(X_test[i].reshape(7,1).T)[:,0], 
      "\n real y =", y_difficult(X_test[i]), 
      "\n predicted tau =", models.mu_uncertainty(X_test[i].reshape(7,1).T),
      "\n predicted sigma =", soft(model.predict(X_test[i].reshape(7,1).T)[:,1]),
      "\n real sigma =", sigma_difficult(X_test[i]),
      "\n confidence interval =", confidence_interval(soft(model.predict(
          X_test[i].reshape(7,1).T)[:,1]).numpy(),
                                    models.sigma_uncertainty(X_test[i].reshape(7,1).T)))


N_tests = 100
N_x_values = 40
results = np.zeros((N_tests, N_x_values))
results2 = np.zeros((N_x_values))
X_testing =  get_fancy_data(N_x_values, 0)[0]
N_dimensions = 7 
for i in range(N_tests):
    
    X_train, Y_train, X_test, Y_test = get_fancy_data(20000, 0)          
    models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 30]), 
                            n_hidden_2 = np.array([50, 50, 30]), n_epochs = 80,
                            n_epochs_2 = 50, verbose = False)
    model = models.model
    for j, x in enumerate(X_testing):
        results[i, j] = (model.predict(x.reshape(N_dimensions,1).T)[:,0] \
                         - y_difficult(x)) \
                        / (models.mu_uncertainty(x.reshape(N_dimensions,1).T)) 
        interval = confidence_interval(soft(model.predict(
                x.reshape(N_dimensions,1).T)[:,1]).numpy(),
                models.sigma_uncertainty(x.reshape(N_dimensions,1).T))
        sigma_real = sigma_difficult(x).numpy()
        if sigma_real > interval[0] and sigma_real < interval[1]:
            results2[j] += 1
    print(i, "/", N_tests)   

for i in range(N_x_values):
    plt.title("sigma = {std}".format(std = np.round(np.std(results, axis = 0)[i],2)))
    plt.hist(results[:,i])
    plt.show()


std = np.round(np.std(results, axis = 0), 2)
stds = results2 / N_tests    

models.mu_uncertainty(0.1 * X_testing[20].reshape(N_dimensions,1).T)

#%% Plots

plt.hist(std)
plt.title('Standard deviations of T')
plt.xlabel("std(T)")
plt.show()

plt.hist(stds)
plt.title('Coverage fractions of confidence interval for $\sigma$')
plt.xlabel('Coverage fraction')
plt.show()


