#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:28:02 2020

@author: paul
"""

#%% Import
import numpy as np
import pandas as pd
from basic_rnn9 import BasicRNN
import keras
import matplotlib.pyplot as plt
#%% Load
data_path = ("/home/paul/Documents/Research/Keras/MNIST/")
test_data = pd.read_csv(data_path + "mnist_test.csv", header=None)
train_data = pd.read_csv(data_path + "mnist_train.csv", header=None)
#%% Initialize
                         
# Prime factors - 784:    1,   2,   4,   7,  8, 14, 16, 28
                    #   784, 392, 196, 112, 98, 56, 49, 28
T = 1       
     
epochs = 1 
batch_size = 600
classes = 10

m = int(784/T)                         
#%% Reshape Data
x_train = train_data.iloc[:,1:np.size(train_data,1)]
x_test = test_data.iloc[:,1:np.size(test_data,1)]

y_train = train_data.iloc[:,0]
y_test = test_data.iloc[:,0]

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], T, m)/255
x_test = x_test.reshape(x_test.shape[0], T, m)/255

z_test = y_test
z_train = y_train

y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

go = BasicRNN(m, 128, classes, e=1, gamma=1, var=1, learning_rate=1)
go.train(epochs, batch_size, x_train, y_train, x_test, y_test)

err, h_n, output, prob_dis, pred, loss, W_hh, V_h, b_h, W_oh = go.get_quants(x_test[0:batch_size], y_test[0:batch_size], z_test[0:batch_size])

plt.plot(err)




