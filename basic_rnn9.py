#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:25:04 2020

@author: paul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = ("/home/paul/Documents/Research/Keras/MNIST/")
weight_path = ("/home/paul/Documents/Research/Keras/MNIST/weights_VANILLA/Row_By_Row/")

class BasicRNN:
### prelim
    def __init__(self, input_dim, n, classes, e, gamma, var, learning_rate):
        super(BasicRNN, self).__init__()
                
        #### RANDOM WEIGHTS
        self.gamma_I = gamma*np.identity(n)
        self.W_oh = np.random.normal(0, var/n, [classes,n])          
        self.W_hh = np.random.normal(0, var/n, [n,n])                 
        self.V_h = np.random.normal(0, 1/input_dim, [n,input_dim])                     
        self.b_h = np.zeros([n,1])                     
 
        #### PRELOADED WEIGHTS      
#        W_oh = np.transpose(pd.read_csv(weight_path + "weights_VANILLA_Woh.csv", header=None))
#        W_hh = np.transpose(pd.read_csv(weight_path + "weights_VANILLA_Whh.csv", header=None))
#        V_h = np.transpose(pd.read_csv(weight_path + "weights_VANILLA_Vh.csv", header=None))
#        b_h = pd.read_csv(weight_path + "weights_VANILLA_bh.csv", header=None)        
#        self.W_oh, self.W_hh, self.V_h, self.b_h = np.array(W_oh), np.array(W_hh), np.array(V_h), np.array(b_h)
        
        self.e = e
        self.n = n
        self.learning_rate = learning_rate
        self.classes = classes
        self.input_dim = input_dim

### train
    def train(self, epochs, batch_size, x_train, y_train, x_test, y_test):
        iterations = int(len(x_train)/batch_size)
        self.err = np.array([])
        for epoch in range(epochs):     
            for iteration in range(iterations):
                x = x_train[batch_size*iteration:batch_size+batch_size*iteration]
                y = y_train[batch_size*iteration:batch_size+batch_size*iteration]
                h_n = self.compute_hidden(x)
                prob_dis = self.forward(h_n)                
                train_acc, train_error = self.get_loss(prob_dis, y)   
                self.err = np.append(self.err, train_error)
                print('Epoch', epoch+1, '/', epochs, ' Training accuracy: ', train_acc)
                print(' Training error: ', train_error, ' Iteration: ', iteration+1)    
                self.update_weights(x, y, h_n)
            x = x_test[0:batch_size]
            y = y_test[0:batch_size]
            h_n = self.compute_hidden(x)    
            prob_dis = self.forward(h_n)                
            test_acc, test_error = self.get_loss(prob_dis, y)         
            print('Epoch', epoch+1, '/', epochs, ' Testing accuracy: ', test_acc)
            print(' Testing error: ', test_error)
        return 

### compute_hidden    
    def compute_hidden_col(self, x_batch_t):
        h_0 = np.zeros([self.n,1])        
        Whh_h0 = np.matmul(self.W_hh,h_0)
#        Whh_h0 = np.matmul(self.W_hh - np.transpose(self.W_hh) - self.gamma_I, h_0)
        x_t = np.expand_dims(x_batch_t[:], axis=1)
        Vh_X = np.matmul(self.V_h,x_t)
        activation = np.tanh(Whh_h0 + Vh_X + self.b_h)
        h_1 = activation
#        h_1 = h_0 + self.e*activation
        h_0 = h_1
        return h_1

    def compute_hidden_page(self, x_batch):        
        h_t = np.zeros([self.n,1])
        time_steps = len(x_batch)        
        for t in range(time_steps):       
            h_1 = self.compute_hidden_col(x_batch[t])
            h_t = np.hstack((h_t,h_1))
        return h_t

    def compute_hidden(self, x):
        time_steps = len(x[0])        
        batch_size = len(x)
        h_n = np.zeros([self.n,time_steps+1])
        for batch in range(batch_size):
            h_t = self.compute_hidden_page(x[batch])
            h_n = np.dstack((h_n,h_t))
        h_n = np.swapaxes(h_n, 0,2)
        h_n = np.swapaxes(h_n, 1,2)
        h_n = np.delete(h_n, [0], axis=0)
        return h_n

### forward        
    def forward(self, h_n):
        output = self.output(h_n)
        prob_dis = self.prob_dis(output)                
        return prob_dis
   
    def output(self, h_n):
        output = np.matmul(self.W_oh, h_n)
        return output 
   
    def prob_dis(self, output):
        prob_dis = self.softmax_ten(output)
        return prob_dis

### predict    
    def predict(self, prob_dis):
        pred = []
        for batch in range(len(prob_dis)):            
            pred = np.append(pred, np.argmax([prob_dis[batch,:,-1]]))
        pred = np.array(pred)    
        return pred     

### get_loss
    def get_loss(self, prob_dis, y):
        batch_size = len(prob_dis)
        correct = 0
        error = 0        
        for batch in range(batch_size):
            y_hat = prob_dis[batch,:,-1]
            error = error - np.dot(y[batch][:], np.log(y_hat))            
            pred = self.predict(prob_dis)
            if np.argmax(y[batch]) - pred[batch]==0:
                correct += 1            
        acc = np.round(correct/batch_size, 4)
        error = np.round(error/batch_size, 3)
        return acc, error 
    
### derivatives
    def J(self, h_n):
        der_act = 1 - h_n*h_n
        J = np.matmul(self.W_hh.T, der_act)
        return J        

    def dh_db(self, h_n):
        der_act = 1 - h_n*h_n
        dh_db = np.expand_dims(np.sum(np.sum(der_act, axis=2), axis=0), axis=1)
        return dh_db
    
### backward
    def backward(self, x_train, y_train, h_n):
        self.correction_Woh = 0  
        self.correction_bh = np.zeros([self.n,1])        
        self.correction_Vh = 0
        self.correction_Whh = 0
        N = len(x_train)
 
        output = np.matmul(self.W_oh, h_n)
        prob_dis = self.softmax_ten(output)
        dE_dWoh = 0
#        dE_dh = 0 
#        Jaco = 0
        for batch in range(N):
            T = len(prob_dis[0][0])
            for t in range(T):
                dE_dWoh += np.outer(prob_dis[batch,:,t] - y_train[batch,:], h_n[batch,:,t])
#                for k in range(t+1,T):
#                    Jaco += self.J(h_n)
#                dE_dh += np.expand_dims(np.matmul((prob_dis[batch,:,t] - y_train[batch,:]).T, self.W_oh), axis=1)
                    
#            self.correction_Whh += dE_dWhh                
#            self.correction_Vh += dE_dVh 
#            self.correction_bh += dE_dbh
            self.correction_Woh += dE_dWoh
#        Jacobian = np.expand_dims(np.sum(np.sum(Jaco, axis=2), axis=0), axis=1)
#        self.correction_bh = dE_dh*Jacobian*self.dh_db(h_n)

        self.correction_Whh = self.correction_Whh/(N*T)
        self.correction_Vh = self.correction_Vh/(N*T)
        self.correction_bh = self.correction_bh/(N*T)
        self.correction_Woh = self.correction_Woh/(N*T)

#        self.correction_Whh = np.clip(self.correction_Whh, -1, 1)
#        self.correction_Vh = np.clip(self.correction_Vh, -1, 1)
#        self.correction_bh = np.clip(self.correction_bh, -1, 1)
#        self.correction_Woh = np.clip(self.correction_Woh, -1, 1)

#        self.correction_bh = np.expand_dims(self.correction_bh, axis=1)

        return self.correction_Woh, self.correction_bh, self.correction_Vh, self.correction_Whh
   
### update weights
    def update_weights(self, x, y, h_n):
        correction_Woh, correction_bh, correction_Vh, correction_Whh = self.backward(x, y, h_n)
#        self.W_hh = self.W_hh - self.learning_rate*correction_Whh
#        self.V_h = self.V_h - self.learning_rate*correction_Vh
        self.b_h = self.b_h - self.learning_rate*correction_bh
        self.W_oh = self.W_oh - self.learning_rate*correction_Woh           
        return

### get_quants
    def get_quants(self, x, y, z):
        W_hh = self.W_hh
        V_h = self.V_h
        b_h = self.b_h
        W_oh = self.W_oh
        err = self.err
        h_n = self.compute_hidden(x) 
        output = self.output(h_n)
        prob_dis = self.prob_dis(output)
        pred = self.predict(prob_dis)
        loss = self.get_loss(prob_dis, y)

        return [err, h_n, output, prob_dis, pred, loss, W_hh, V_h, b_h, W_oh] #, cor_W_hh, cor_V_h, cor_b_h, cor_W_oh]
    
### softmax
    def softmax(self, x):
        p = np.exp(x-np.max(x))
        return p / np.sum(p)

    def softmax_mat(self, x):
        soft = self.softmax(x[:,0])
        soft2 = self.softmax(x[:,1])
        soft_f = np.vstack((soft,soft2))
        for col in range(2,len(x[0])):
            soft = self.softmax(x[:,col])
            soft_f = np.vstack((soft_f,soft))            
        return soft_f.T

    def softmax_ten(self, x):
        soft = self.softmax_mat(x[0])
        soft2 = self.softmax_mat(x[1])
        soft_f = np.dstack((soft,soft2))
        for pg in range(2,len(x)):
            soft = self.softmax_mat(x[pg])
            soft_f = np.dstack((soft_f,soft))  
        soft_f = np.swapaxes(soft_f, 0, 2)
        soft_f = np.swapaxes(soft_f, 1, 2)       
        return soft_f
