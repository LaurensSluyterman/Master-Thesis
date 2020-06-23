"""
In this script we compare a bayesian linear model with a one neuron
neural network. We compare the exact posterior with the approximate 
posterior. Note that the file gal.py contains the code to train 
network with dropout. 
"""

import numpy as np
from matplotlib import pyplot as plt
#%% Creating data


def y(x):
    """Define the mean of the data."""
    return x + 2 

def sigma(x):
    """Define the standard deviation of the data."""
    return 1

N_datapoints = 3000 
N_test = 2000
X_train = np.array(np.linspace(-2,2,N_datapoints)) 
Y_train = np.zeros(N_datapoints)
X_test = np.array(np.linspace(-1.9,1.9,N_test)) 
Y_test = np.zeros(N_test)
real_train = np.zeros(N_datapoints)
real_test = np.zeros(N_test)

for i in range(0,N_datapoints):
    Y_train[i]=y(X_train[i])+np.random.normal(0,sigma(X_train[i]))
    real_train[i]=y(X_train[i])

for i in range(0, N_test):
    Y_test[i]=y(X_test[i])+np.random.normal(0,sigma(X_test[i]))
    real_test[i]=y(X_test[i])
    

#%% Training a model using Gal's code (found in Gal.py)
a = np.ones(len(X_train))
X_train2 = np.stack((a, X_train), axis = -1)
b = np.ones(len(X_test))
X_test2 = np.stack((b, X_test), axis = -1)
model = net(X_train2, Y_train, n_hidden = np.array([1])).model

#%% Exact linear Bayesian model
a = np.ones(len(X_train))
Phi = np.stack((a, X_train), axis = -1)
alpha = 1.0
beta = 1 
Sinv = alpha * np.identity(2) + beta * np.transpose(Phi).dot(Phi)
S = np.linalg.inv(Sinv)
m = beta * S.dot(np.transpose(Phi)).dot(Y_train)

# We weights of the linear model have the following distribution
weights = np.random.normal(loc = m, scale = np.abs(S))
# Dropout gives the following weights
model.weights()


#We compare S to the following
0.05 * (1 - 0.05) * model.weights[0].numpy() ** 2 * np.identity(2)

#%% Plots

#Visualize the test data
plt.title("Test Data")
plt.plot(X_train,Y_train, 'o')
plt.plot(X_train,real_train)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Visualize the results from MC-dropout.
plt.plot(X_test, model.predict(X_test2), 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(X_test,Y_test)
plt.show()