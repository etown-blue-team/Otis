#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

import numpy as np
from matplotlib import pyplot as plt
input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))  #Takes an input number and puts it between 1 and 0
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]]) #Our inputs
labels = np.array([[1,0,0,1,1]]) #The Outputs
labels = labels.reshape(5,1)
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)

lr =0.05 #Learning Rate

for epoch in range(20000): #20,000 Runs of Training Data

    #Feedforward
    XW = np.dot(feature_set, weights) + bias  #Input dot Weight · Bias  θx,w = 1/(1+e^(-X·W))
    z = sigmoid(XW) #Put it between 1 and 0 for outputs, Z is our predicted output

    #Backpropigation
    error = z - labels #z - lables will give us how far off the output is, if its spot on it'll be around 0
    print(error.sum()) #Show the used

    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num


single_point = np.array([0,1,0])
result = sigmoid(np.dot(single_point,weights) + bias)
print(result)