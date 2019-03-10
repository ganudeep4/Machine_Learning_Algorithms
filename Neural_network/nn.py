#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs



def calculate_loss(model, X, y):
    w1, b1 = model.get('w1'), model.get('b1')
    w2, b2 = model.get('w2'), model.get('b2')
    
#     Calculating loss using forward propogation
    a = np.dot(X, w1) + b1
    h = np.tanh(a)
    z = np.dot(h, w2) + b2
    y_pred = np.exp(z)
    sum_y_pred = np.sum(y_pred, axis=1, keepdims=True)
    y_pred = y_pred/sum_y_pred
    
    loss = np.sum(np.sum(y*np.log(y_pred), axis=0))
    loss = -(loss/y.shape[0])
        
    return loss


def predict(model, x):
    w1, b1 = model.get('w1'), model.get('b1')
    w2, b2 = model.get('w2'), model.get('b2')
    
#     Calculating output of X
    a = np.dot(x, w1) + b1
    h = np.tanh(a)
    z = np.dot(h, w2) + b2
        
    y_pred = np.exp(z)
    y_pred = y_pred/np.sum(y_pred)
    
    return np.argmax(y_pred, axis=1)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
#     Initialization of weights and biases of the network
    W1 = np.random.randn(np.size(X, 1), nn_hdim)    
    b1 = np.random.randn(1, nn_hdim)
    W2 = np.random.randn(nn_hdim, np.size(X, 1))
    b2 = np.random.randn(1, np.size(X, 1))
    
    model = {}
    iteration = 1
    
     # Reshaping y in 2D
    temp = np.zeros((y.shape[0], X.shape[1]))
    for i in range(y.shape[0]):
        if y[i] == 0:
            temp[i][0] = 1
        else:
            temp[i][1] = 1
    y = temp
        
    while iteration <= num_passes:
        
    #     Forward propogation
        a = np.dot(X, W1) + b1
        h = np.tanh(a)
        z = np.dot(h, W2) + b2
        y_pred = np.exp(z)
        sum_y_pred = np.sum(y_pred, axis=1, keepdims=True)
        y_pred = y_pred/sum_y_pred

    #     Backward propogation
        dL_dy_pred = y_pred - y

        dL_da = (1-(np.tanh(a)**2))   
        temp = np.dot(dL_dy_pred, W2.transpose())    
        dL_da = dL_da * temp

        dL_dW2 = np.dot(h.transpose(),dL_dy_pred)
        dL_db2 = np.sum(dL_dy_pred, axis=0)
        dL_dW1 = np.dot(X.transpose(), dL_da)
        dL_db1 = np.sum(dL_da, axis=0)

    #     Gradient descent updating
        W1 -= 0.001 * dL_dW1
        b1 -= 0.001 * dL_db1
        W2 -= 0.001 * dL_dW2
        b2 -= 0.001 * dL_db2

        model['w1'] = W1
        model['b1'] = b1
        model['w2'] = W2
        model['b2'] = b2
        
        iteration += 1
        
        if print_loss:
            if iteration % 1000 == 0:
                print('Loss at %d iteration is %f' % (iteration, calculate_loss(model, X, y)*100))
    
    return model


def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
    output_layer_neurons = 3 
#     Initialization of weights and biases of the network
    W1 = np.random.randn(np.size(X, 1), nn_hdim)    
    b1 = np.random.randn(1, nn_hdim)
    W2 = np.random.randn(nn_hdim, output_layer_neurons)
    b2 = np.random.randn(1, output_layer_neurons)
    

    model = {}
    iteration = 1
    
     # Reshaping y in 2D
    temp = np.zeros((y.shape[0], output_layer_neurons))
    for i in range(y.shape[0]):
        if y[i] == 0:
            temp[i][0] = 1
        elif y[i] == 1:
            temp[i][1] = 1
        else:
            temp[i][2] = 1
    y = temp
        
    while iteration <= num_passes:
        
    #     Forward propogation
        a = np.dot(X, W1) + b1
        h = np.tanh(a)
        z = np.dot(h, W2) + b2
        y_pred = np.exp(z)
        sum_y_pred = np.sum(y_pred, axis=1, keepdims=True)
        y_pred = y_pred/sum_y_pred

    #     Backward propogation
        dL_dy_pred = y_pred - y

        dL_da = (1-(np.tanh(a)**2))   
        temp = np.dot(dL_dy_pred, W2.transpose())    
        dL_da = dL_da * temp

        dL_dW2 = np.dot(h.transpose(),dL_dy_pred)
        dL_db2 = np.sum(dL_dy_pred, axis=0)
        dL_dW1 = np.dot(X.transpose(), dL_da)
        dL_db1 = np.sum(dL_da, axis=0)

    #     Gradient descent updating
        W1 -= 0.001 * dL_dW1
        b1 -= 0.001 * dL_db1
        W2 -= 0.001 * dL_dW2
        b2 -= 0.001 * dL_db2

        model['w1'] = W1
        model['b1'] = b1
        model['w2'] = W2
        model['b2'] = b2
        
        iteration += 1
        
        if print_loss:
            if iteration % 1000 == 0:
                print('Loss at %d iteration is %f' % (iteration, calculate_loss(model, X, y)*100))
    
    return model

    