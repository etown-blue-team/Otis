import numpy as np
from network import *

data = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([[0],[1],[1],[0]])

neural = Network(data,output)
Network.train(neural,1)
# Network.run(neural,[0,0])

