# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
#%% Creating data

const=0.3
def y(x):
    return x + 2 

def sigma(x):
    return 1

N = 10000  #I generate 100.000 datapoint
Ntest = 2000
Xtrain=np.array(np.linspace(-2,2,N)) #In the interval [-2,2]
Ytrain=np.zeros(N)
Xtest=np.array(np.linspace(-1.9,1.9,Ntest)) #In the interval [-2,2]
Ytest=np.zeros(Ntest)
real=np.zeros(N)
real2=np.zeros(Ntest)
for i in range(0,N):
    Ytrain[i]=y(Xtrain[i])+np.random.normal(0,sigma(Xtrain[i]))
    real[i]=y(Xtrain[i])

for i in range(0, Ntest):
    Ytest[i]=y(Xtest[i])+np.random.normal(0,sigma(Xtest[i]))
    real2[i]=y(Xtest[i])
plt.title("Test Data")
plt.plot(Xtrain,Ytrain)
plt.plot(Xtrain,real)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%% Training a model using Gal's code
a = np.ones(len(Xtrain))
Xtrain2 = np.stack((a, Xtrain), axis = -1)
b = np.ones(len(Xtest))
Xtest2 = np.stack((b, Xtest), axis = -1)
model = net(Xtrain2, Ytrain, n_hidden = np.array([1])).model
plt.plot(Xtest, model.predict(Xtest2), 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(Xtest,Ytest)
plt.show()
model.weights

#%% Exact linear Bayesian model
a = np.ones(len(Xtrain))
Phi = np.stack((a, Xtrain), axis = -1)
alpha = 1.0
beta = 1 
Sinv = alpha * np.identity(2) + beta * np.transpose(Phi).dot(Phi)
S = np.linalg.inv(Sinv)
m = beta * S.dot(np.transpose(Phi)).dot(Ytrain)

weights = np.random.normal(loc = m, scale = np.abs(S))

0.05 * (1 - 0.05) * model.weights[0].numpy()**2 * np.identity(2)

#%%
l = []
for i in range(0, 100):
    l.append(model.predict(np.array([[0,1]])))
