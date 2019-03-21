#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

import numpy as np
input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))  #Takes an input number and puts it between 1 and 0
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Smoker, Obese, Excersize, Causing causing Attr 1, Not Causing Attr 2



feature_set = np.array([[0,0,1,0,0],[1,1,0,0,0],[0,1,0,0,0],[1,0,0,1,0],[0,0,1,0,1],[1,0,0,0,0],[1,1,0,1,0],[0,1,0,0,0],[1,1,0,1,0],[0,1,0,0,0],[0,0,1,1,0],[1,1,0,0,1],[1,0,0,0,1],[0,1,0,0,0],[1,0,0,0,0],[1,0,1,0,1],[1,0,0,1,0],[0,1,0,0,1],[0,1,0,0,0],[1,1,0,1,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,1,0],[0,1,1,0,0],[1,0,0,0,1],[1,1,0,0,0],[1,0,1,0,0]]) #Our inputs
labels = np.array([0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0]) #The Outputs


labels = labels.reshape(27,1)
np.random.seed(42)
weights = np.random.rand(5,1)
bias = np.random.rand(1)

lr =0.05 #Learning Rate

for epoch in range(20000): #20,000 Runs of Training Data

    #Feedforward
    XW = np.dot(feature_set, weights) + bias  #Input dot Weight Bias x,w = 1/(1+e^(-XW))
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


data = np.array([0,0,0,0,0])
print(type(data))
print(type(weights))
result = sigmoid(np.dot(data,weights))
print(result)
