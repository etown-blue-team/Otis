import numpy
from Network import *

data = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1]])
output = np.array([0,0,1,1,1,1,0,0])

nerual = Network(data,output)
Network.train(nerual,500)
Network.run(nerual,[0,0])

