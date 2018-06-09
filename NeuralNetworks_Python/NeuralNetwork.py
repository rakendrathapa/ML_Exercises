# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        
        # Setting the number of nodes in layer 2 and layer 3 
        # more nodes.
        l2 = 5
        l3 = 4
        
        # Assign random weights to matrices in the network
        # Format is (no. of nodes in previous layer) x (no. of nodes in the following layer)
        self.synaptic_weights1 = 2 * random.random((3, l2)) - 1
        self.synaptic_weights2 = 2 * random.random((l2, l3)) - 1
        self.synaptic_weights3 = 2 * random.random((l3, 1)) - 1
        
    def __sigmoid(self, x):
        return 1/(1+exp(-x))
    
    # Derivative of sigmoid function, indicates the confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x*(1-x)
    
    #

