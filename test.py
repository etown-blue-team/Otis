import numpy as np
from network import *

# data = np.array([[0,0],[0,1],[1,0],[1,1]])
# output = np.array([[0],[1],[1],[0]]) # XOR
# output = np.array([[0],[1],[1],[1]]) # OR

data = np.array([[0,0,1],[0,1,1],[1,0,1],[0,1,0],[1,0,0],[1,1,1],[0,0,0]])
output = np.array([[0],[1],[1],[1],[1],[0],[0]])



neural = Network(data, output)
Network.train(neural, 10000)
Network.run(neural,[1,1,0])