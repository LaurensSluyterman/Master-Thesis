"""
Created on Sat May 30 14:20:41 2020.

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
#%%
def p(x):  
    """Return the probabilty of 1 given x."""
    return 0.7 * np.exp(-x ** 2)



def get_classification_data(N_train, N_test):  
    """
    Create a dataset consisting if 0/1 entries.
    
    This function creates a dataset containing the N_train training samples
    and N_test testing samples genereated according to p(x). For every
    x, y has value 1 with probability p(x) and zero with probability
    1 - p(x).
    
    Parameters:
        N_train (int): The number of training samples 
        N_test  (int): The number of test samples
    
    Returns:
        X_train, Y_train, X_test, Y_test: arrays genereated using p(x).
        
    """  
    X_train = np.array(np.linspace(-1,1,N_train)) 
    Y_train = np.random.binomial(1, p(X_train))
    X_test = np.array(np.linspace(-0.999,0.999, N_test))
    Y_test = np.random.binomial(1, p(X_test))

    return X_train, Y_train, X_test, Y_test

def soft(x):
    """Return log(1 + exp(x)) + 1e-6."""
    return 1e-6 + tf.math.softplus(x)



class Neural_network: 
    """
    This class represents a model with extra networks to give uncertainties.
    
    In this class a total of 7 models are trained. The first model is 
    trained on the entire dataset and is used as our predictor. This model
    outputs p_hat and is trained using the loglikelihood of the data assuming 
    a binomial distribution. 
    In order to get an estimate of the uncertainty of p_hat,
    6 more models are trained. First the data is split in three equal parts,
    then three more models are trained on these parts (with the same 
    structure as the main model). Then these 3 models are used to produce
    new targets hat(p)_seen, hat(p)_unseen,for each pair that we can make by 
    picking two models out of the three. This is done by letting the two model 
    first predict on the  part of the data they were trained on and then on 
    the other part. 
    Using these new targets an additional three models are trained to obtain
    uncertainty estimates for p_hat.
   
    Attributes:
        model: The trained network on all of the data, it outputs p_hat.
        model_tau_i: A network trained to predict the uncertainty in p_hat
    
    
    Methods:
       p_uncertainty: Gives the predicted uncertainty of p_hat at a given
                value of x.
                
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
        input_shape = 1
        X_train_1, X_train_2, Y_train_1, Y_train_2 = train_test_split(
                    X_train, Y_train, test_size = 2 / 3)
        X_train_2, X_train_3, Y_train_2, Y_train_3 = train_test_split(
                    X_train_2, Y_train_2, test_size = 0.5)
        
        def loss(targets, outputs):
            """
            Give the loss function used for the main model.
            
            This function calculates the negative loglikelihood of a
            binomial distribution with probabilty p(x).
            """
            p = (outputs[...,0:1])
            y = targets[...,0:1]
            l = y * K.log(p) + (1 - y) * K.log(1 - p)
            return - l
        
        def loss_tau(targets, outputs):
            """
            Give the loss function used for the tau-models.
            
            This function calculates the negative loglikelihood of a 
            normal distribution with mean p_hat and standard deviation tau.
            """
            tau = soft(outputs)
            l = -K.log(tau) - 0.5 * K.square((targets[...,:1] - 
                                              targets[...,1:])  / tau)
            return  - l
            

        
        def get_model(n_hidden, c, loss, n_out, final_activation = 'linear'):
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
            outputs = Dense(n_out, activation = final_activation)(inter)
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
            p_own = np.vstack((model_a.predict(X_a), 
                                model_b.predict(X_b)))[:,0]
            p_other = np.vstack((model_b.predict(X_a), 
                                model_a.predict(X_b)))[:,0]
            tau_targets = np.stack((p_own, p_other), axis = 1)
            return tau_targets

        model = get_model(n_hidden, 1 / len(X_train), loss, 1,
                          final_activation = 'sigmoid')
        model_1 = get_model(n_hidden, 1/ len(X_train_2), loss, 1,
                            final_activation = 'sigmoid')
        model_2 = get_model(n_hidden, 1 / len(X_train_2), loss, 1,
                            final_activation = 'sigmoid')
        model_3 = get_model(n_hidden, 1 / len(X_train_2), loss, 1,
                            final_activation = 'sigmoid')
        model_tau_1 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        model_tau_2 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        model_tau_3 = get_model(n_hidden, 3 / (2 * len(X_train)), loss_tau, 1)
        
        model.fit(X_train, Y_train, batch_size = 100, epochs = n_epochs, 
                  verbose = verbose)
        model_1.fit(X_train_1, Y_train_1, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        model_2.fit(X_train_2, Y_train_2, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        model_3.fit(X_train_3, Y_train_3, batch_size = 100, epochs = n_epochs,
                    verbose = verbose)
        
            
        tau_12_targets = get_targets(model_1, model_2, 
                                                    X_train_1, X_train_2)
        tau_13_targets = get_targets(model_1, model_3,
                                                    X_train_1, X_train_3)
        tau_23_targets = get_targets(model_2, model_3, 
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

        self.model = model   
        self.model_tau_1 = model_tau_1
        self.model_tau_2 = model_tau_2
        self.model_tau_3 = model_tau_3

    def p_uncertainty(self, x): 
        """
        Give the predicted uncertainty of mu_hat at x.
        
        This function calculates the average of the predictions of the 
        three tau_models that were previously trained. 
        """
        tau_1 = soft(self.model_tau_1.predict(x)).numpy()[0]
        tau_2 = soft(self.model_tau_2.predict(x)).numpy()[0]
        tau_3 = soft(self.model_tau_3.predict(x)).numpy()[0]
        return np.max([tau_1, tau_2, tau_3])
        
         
       
#%% Testing
    
# Test an indidual simulation for 1 value of x
X_train, Y_train, X_test, Y_test = get_classification_data(20000, 1000)                
models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 20]),
                        n_hidden_2 = np.array([50, 50, 20]),n_epochs = 20, n_epochs_2 = 20)

model = models.model

# Varying x to large values demonstrates the oultier detection ability.
x = np.array([100])
print(" predicted p =", model.predict(x)[:,0], 
      "\n real p =", p(x), 
      "\n predicted tau =", models.p_uncertainty(x))

# Test a number of simulations for multiple values of x
N_tests = 100
N = 40
x_tests = np.linspace(-1, 1, N)
results = np.zeros((N_tests, N))

for i in range(N_tests):
    
    X_train, Y_train, X_test, Y_test = get_classification_data(10000, 0)          
    models = Neural_network(X_train, Y_train, n_hidden = np.array([50, 50, 30]), 
                            n_hidden_2 = np.array([50, 50, 30]), n_epochs = 30,
                            n_epochs_2 = 30, verbose = False)
    model = models.model
    for j, x in enumerate(x_tests):
        x = np.array([x])
        results[i, j] = (model.predict(x)[:,0] - p(x)) / (
                models.p_uncertainty(x))
    print(i, "/", N_tests)   



#%% Plots

std_T = np.round(np.std(results, axis = 0), 2)

for i in range(N):
    plt.title("x = {x}, sigma = {std}".format(x = np.round(x_tests[i], 2), 
                                              std = np.round(np.std(results, 
                                                                    axis = 0)[i],2)))
    plt.hist(results[:,i])
    plt.show()


plt.plot(X_test, p(X_test), label = 'p')
plt.plot(X_test, model.predict(X_test)[:,0], label = "$\hat{p}$")
plt.xlabel("$x$")
plt.ylabel("p(x)")
plt.title('Predicted $p$')
plt.legend()
plt.show()


plt.plot(X_test, p(X_test))
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Training data")
plt.show()

plt.plot(x_tests, std_T)
plt.xlabel("x")
plt.ylabel("Std(T)")
plt.ylim([0, 2])
plt.title("Standard deviation of T")
plt.show()

