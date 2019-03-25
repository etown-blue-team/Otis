from sklearn import datasets  
import numpy as np  
import matplotlib.pyplot as plt

np.random.seed(0)  
feature_set, labels = datasets.make_moons(5, noise=0.10)  

labels = labels.reshape(5, 1)

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

wh = np.random.rand(len(feature_set[0]),4)  #2x4
wo = np.random.rand(4, 1)  
lr = 0.5

print("FS") #5x2 input data, 5 rows of data 2 inputs each
print(feature_set)
print("FSL") #2
print(len(feature_set[0]))
for epoch in range(1):  
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)
    print("WH") #2x4 each row is all of the weights from 1 input
    print(wh)
    print("ZH") #5x4
    print(zh)
    print("AH") #5x4
    print(ah)
    print("WO") #4x1
    print(wo)
    print("ZO") #4x1
    print(zo)
    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    # print(error_out.sum())

    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ================

    wh -= lr * dcost_wh
    wo -= lr * dcost_wo