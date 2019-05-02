import numpy as np
import math

class Network:
    '''
    Initalizes the network with the input and output data

    Parameters:
        inData: the input data, the data being tested against
        outData: the output data, the result expected
    '''
    def __init__(self, inData, outData):
        self.inData = inData
        self.outData = outData
        self.hiddenLayers = 2 # The user defined numbers of hidden layers to build into the network
        self.trainedInputWeights = []
        self.trainedHiddenWeights = []
        self.trainedOutputWeights = []


    '''
    The sigmoid function, puts everything to be a probability between 0 and 1

    Paramaters:
        x: the number to be run through the sigmoid
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

    '''
    Derivative of the sigmoid function

    Parameters:
        x: the number to be run through the sigmoid derivative
    '''
    def sigmoid_der(self, x):
        return x * (1 - x)


    '''
    Softmax function, takes a number and normalizes it into a probability

    Parameters:
        x: the number to run through the softmax function
    '''
    def softmax(self, x):
        expX = np.exp(x)
        return expX / expX.sum()

    # Rectified Linear Unit Activiation Function
    # Not working yet
    # def sigmoid(self, x): #This is just for quick testing
    # def relu(self, x):
        # arr = np.zeros((x.shape))
        # for i in range(0, len(x) - 1):
        #     for j in range(0 , len(x[0]) - 1):
        #         if x[i][j] > 0:
        #             arr[i][j] = 1
        # print(arr)
        # return arr

    '''
    Training function

    This is where the magic happens

    Parameters:
        epochs: the number of epochs to run
    '''
    def train(self, epochs):
        nodesInput = len(self.inData[0])
        nodesOutput = len(self.outData[0])
        nodesHidden = math.ceil((nodesInput + nodesOutput) / 2) # Mean of the I/O nodes rounded up

        # Returns random values between [0,1) to a given shape
        inputWeights = np.random.rand(nodesInput, nodesHidden)
        hiddenWeights = np.random.rand(self.hiddenLayers - 1, nodesHidden, nodesHidden)
        outputWeights = np.random.rand(nodesHidden, nodesOutput)

        for epoch in range(epochs):
            # =====[ Feed Forward ]=====

            # Input -> Hidden
            inputLayerNodeValues = self.sigmoid(np.dot(self.inData, inputWeights))

            # Hidden -> Hidden
            if self.hiddenLayers > 1:
                hiddenLayerNodeValues = []
                hiddenLayerNodeValues.append(self.sigmoid(np.dot(inputLayerNodeValues, hiddenWeights[0])))

                for i in range(1, self.hiddenLayers - 1):
                    hiddenLayerNodeValues.append(self.sigmoid(np.dot(hiddenLayerNodeValues[i - 1], hiddenWeights[i])))

            # Hidden -> Output
            if self.hiddenLayers > 1:
                outputLayerWeightSums = np.dot(hiddenLayerNodeValues[-1], outputWeights)
            else:
                outputLayerWeightSums = np.dot(inputLayerNodeValues, outputWeights)
            outputLayerNodeValues = self.sigmoid(outputLayerWeightSums)

            # =====[ Back Propagation ]=====

            # Output -> Hidden
            outputError = self.outData - outputLayerNodeValues
            outputDelta = outputError * self.sigmoid_der(outputLayerNodeValues)

            # Hidden -> Hidden
            if self.hiddenLayers > 1:
                hiddenDelta = []
                hiddenError = np.dot(outputDelta, outputWeights.T)
                hiddenDelta.append(hiddenError * self.sigmoid_der(hiddenLayerNodeValues[-1]))

                for i in range(self.hiddenLayers - 3, -1, -1):
                    hiddenError = np.dot(hiddenDelta[0], hiddenWeights[i + 1])
                    hiddenDelta.insert(0, hiddenError * self.sigmoid_der(hiddenLayerNodeValues[i]))

            # Hidden -> Input
            if self.hiddenLayers > 1:
                inputError = np.dot(hiddenDelta[0], hiddenWeights[0].T)
            else:
                inputError = np.dot(outputDelta, outputWeights.T)
            inputDelta = inputError * self.sigmoid_der(inputLayerNodeValues)
           
            # Calculate the adjustment to the weights
            inputAdj = np.dot(self.inData.T, inputDelta)
            if self.hiddenLayers > 1:
                hiddenAdj = []
                hiddenAdj.append(np.dot(inputLayerNodeValues.T, hiddenDelta[0]))
                for i in range(1, self.hiddenLayers - 1):
                    hiddenAdj.append(np.dot(hiddenLayerNodeValues[i - 1].T, hiddenDelta[i]))
                outputAdj = np.dot(hiddenLayerNodeValues[-1].T, outputDelta)
            else:
                outputAdj = np.dot(inputLayerNodeValues.T, outputDelta)

            # Update the Weights
            inputWeights += inputAdj
            for i in range(self.hiddenLayers - 1):
                hiddenWeights[i] += hiddenAdj[i]
            outputWeights += outputAdj

        # Store the weights
        self.trainedInputWeights = inputWeights
        self.trainedHiddenWeights = hiddenWeights
        self.trainedOutputWeights = outputWeights


    '''
    This runs data against the already trained network

    Parameters:
        data: the data to run against the network.

    Returns: the output of the network
    '''
    def run(self, data):
        firstLayer = self.sigmoid(np.dot(data, self.trainedInputWeights))
        if self.hiddenLayers > 1:
            hiddenLayer = self.sigmoid(np.dot(firstLayer, self.trainedHiddenWeights[0]))
            for i in range(1, self.hiddenLayers - 1):
                hiddenLayer = self.sigmoid(np.dot(hiddenLayer, self.trainedHiddenWeights[i]))
            outputLayer = self.sigmoid(np.dot(hiddenLayer, self.trainedOutputWeights))
        else:
            outputLayer = self.sigmoid(np.dot(firstLayer, self.trainedOutputWeights))


        output = np.array_str(outputLayer, precision = 2, suppress_small = True)
        return output