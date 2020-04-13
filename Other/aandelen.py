#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:29:30 2019

@author: Laurens Sluyterman
"""

import pandas_datareader as pdr
from datetime import date
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


def predictor(NAME):
    today = date.today()
    API_KEY = 'XXXXXXXXXXXXXXXX'
    
    DATE_START = '2015-01-01'
    DATE_END = today
    SYMBOL = NAME
    
    # We use the 'av-daily' DataReader to download data from AlphaVantage
    stock_raw = pdr.DataReader(SYMBOL, 'av-daily',
                           start=DATE_START,
                           end=DATE_END,
                           api_key=API_KEY)
    stock = stock_raw
    stock['opendifference'] = 100 * (stock.open.shift(-1) - stock.close) / stock.close
    stock['opendifference1'] = 100 * (stock.open - stock.close.shift(1)) / stock.close.shift(1)
    stock['open1'] = stock.open.shift(1)
    stock['open2'] = stock.open.shift(2)
    #stock['open3'] = stock.open.shift(3)
    #stock['open4'] = stock.open.shift(4)
    #stock['open5'] = stock.open.shift(5)
    #stock['open6'] = stock.open.shift(6)
    
    stock['high1'] = stock.high.shift(1)
    stock['high2'] = stock.high.shift(2)
    #stock['high3'] = stock.high.shift(3)
    #stock['high4'] = stock.high.shift(4)
    #stock['high5'] = stock.high.shift(5)
    #stock['high6'] = stock.high.shift(6)
    
    stock['low1'] = stock.low.shift(1)
    stock['low2'] = stock.low.shift(2)
    #stock['low3'] = stock.low.shift(3)
    #stock['low4'] = stock.low.shift(4)
    #stock['low5'] = stock.low.shift(5)
    #stock['low6'] = stock.low.shift(6)
    
    stock['close1'] = stock.close.shift(1)
    stock['close2'] = stock.close.shift(2)
    #stock['close3'] = stock.close.shift(3)
    #stock['close4'] = stock.close.shift(4)
    #stock['close5'] = stock.close.shift(5)
    #stock['close6'] = stock.close.shift(6)
    
    stock['volume1'] = stock.volume.shift(1)
    stock['volume2'] = stock.volume.shift(2)
    #stock['volume3'] = stock.volume.shift(3)
    #stock['volume4'] = stock.volume.shift(4)
    #stock['volume5'] = stock.volume.shift(5)
    #stock['volume6'] = stock.volume.shift(6)
    stock = stock.dropna()
   # x_star = (stock.iloc[-1] - stock.mean()) / stock.std()
    means = stock.mean()
    means['opendifference'] = 0
    stdevs = stock.std()
    stdevs['opendifference'] = 1
    stock = (stock - means) / stdevs
    
    split = np.int(len(stock) * 0.2)
    train, test = stock.iloc[split + 1:], stock.iloc[0:split]
    
    Xtrain = train.drop(columns=['opendifference'])
    Ytrain = train['opendifference']
    #Ytrain2 = train['opendifference']
    #Ytrain2[Ytrain > 0] = 1
    #Ytrain2[Ytrain < 0] = 0
    Xtest = test.drop(columns=['opendifference'])
    Ytest = test['opendifference']
    #Ytest2 = test['opendifference']
    #Ytest2[Ytest2 > 0] = 1
    #Ytest2[Ytest2 < 0] = 0
   
    
    
    inputs = tf.keras.Input(shape=16)
    c1 = layers.Dense(400, activation = 'relu')(inputs)
    d1 = layers.Dropout(0.1)(c1,training=True)
    c2 = layers.Dense(400, activation='relu')(d1)
    d2 = layers.Dropout(0.1)(c2,training=True)
    output = layers.Dense(1)(d2)
    model = tf.keras.Model(inputs, output)

    
    model.compile(
            loss = 'mse',
            optimizer ='nadam'
            )
    
    model.fit(
                Xtrain, Ytrain,\
                epochs = 10, validation_data = (Xtest, Ytest),\
                verbose = 0, shuffle = True, batch_size = 20)
    
    
    
    
    return model

# 'RSW27TLTQRSOI13W'
def stockdata(NAME):
    today = date.today()
    API_KEY = 'XXXXXXXXXXXXXXXX'
    
    DATE_START = '2019-11-20'
    DATE_END = today
    SYMBOL = NAME
    stock = pdr.DataReader(SYMBOL, 'av-daily',
                           start=DATE_START,
                           end=DATE_END,
                           api_key=API_KEY)
    
    stock['opendifference1'] = 100 * (stock.open - stock.close.shift(1)) / stock.close.shift(1)
    stock['open1'] = stock.open.shift(1)
    stock['open2'] = stock.open.shift(2)
    
    stock['high1'] = stock.high.shift(1)
    stock['high2'] = stock.high.shift(2)

    stock['low1'] = stock.low.shift(1)
    stock['low2'] = stock.low.shift(2)

    stock['close1'] = stock.close.shift(1)
    stock['close2'] = stock.close.shift(2)

    stock['volume1'] = stock.volume.shift(1)
    stock['volume2'] = stock.volume.shift(2)

    stock = stock.dropna()
    means = stock.mean()
    stdevs = stock.std()
    stock = (stock - means) / stdevs
    
    x_star = stock.tail(1)
    
    return x_star
    
def prediction(NAME):
    
    model = predictor(NAME)
    
    def mean_of_passes(X, N_passes):
        Mean = model.predict(X)[:,0]
        for i in range(N_passes):
            Mean += model.predict(X)[:,0]
        return Mean / (N_passes + 1)


    def std_of_passes(X, N_passes):
        varss = []
        for i in range(N_passes):
            l = model.predict(X)[:,0]
            varss.append(l)
        z = np.std(varss,axis=0)
        return z 
    
    print(NAME, \
          'mean prdiction',\
          mean_of_passes(stockdata(NAME), 50),\
          'std of prediction',\
          std_of_passes(stockdata(NAME), 50))