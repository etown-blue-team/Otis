import numpy as np
import math

class Network:
    def __init__(self, inData, outData):
        # TODO:check to make sure that indata/outdata match # of rows
        self.inData = inData
        self.outData = outData
        self.trainedHidden = []
        self.trainedOutput = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        expX = np.exp(x)
        return expX / expX.sum()

    def train(self, epochs):
        nodesInput = len(self.inData[0]) # will likely have to more complex than this
        nodesOutput = 1 # determine if multiple outputs or  before this and adjust accordingly
        nodesHidden = math.ceil((nodesInput + nodesOutput) / 2) # Mean of the I/O nodes rounded up
        hiddenLayers = 2 # determine how many hidden layers the user wants, default to 1

        # Total number of weights = nodesInput * nodesHidden + nodesOutput * nodesHidden + nodesHidden * hiddenLayers
        np.random.seed(100) # temporary
        # Returns random values between [0,1) to a given shape
        inputWeights = np.random.rand(nodesInput, nodesHidden) #2x2 each row is the input weights of the hidden layer
        hiddenWeights = np.random.rand(hiddenLayers - 1, nodesHidden, nodesHidden) #2x2x2
        outputWeights = np.random.rand(nodesHidden, nodesOutput) #2x1
        
        learningRate = 0.5

        # print("IW")
        # print(inputWeights)
        # print("HW")
        # print(hiddenWeights)
        # print("OW")
        # print(outputWeights)

        for epoch in range(epochs):
            # ===[ Feed Forward ]===

            # Input -> Hidden
            inputLayerNodeValues = self.sigmoid(np.dot(self.inData, inputWeights))

            # print("FLNV")
            # print(inputLayerNodeValues)

            # Hidden -> Hidden
            if hiddenLayers > 1:
                hiddenLayerNodeValues = []
                hiddenLayerNodeValues.append(self.sigmoid(np.dot(inputLayerNodeValues, hiddenWeights[0])))

                for i in range(1, hiddenLayers - 1):
                    hiddenLayerNodeValues.append(self.sigmoid(np.dot(hiddenLayerNodeValues[i - 1], hiddenWeights[i])))
            
                # print("HLNV")
                # print(hiddenLayerNodeValues)

            # Hidden -> Output
            if hiddenLayers > 1:
                outputLayerWeightSums = np.dot(hiddenLayerNodeValues[-1], outputWeights)
            else:
                outputLayerWeightSums = np.dot(inputLayerNodeValues, outputWeights)
            outputLayerNodeValues = self.sigmoid(outputLayerWeightSums)

            # print("OLNV")
            # print(outputLayerNodeValues)

            # ===[ Back Propagation ]===
            errorOut = np.power((outputLayerNodeValues - self.outData), 2)
            meanSquaredError = errorOut.sum() / len(self.inData) # Mean squared error cost function
            print(meanSquaredError)

            # Output -> Hidden
            outputError = self.outData - outputLayerNodeValues
            outputDelta = outputError * self.sigmoid_der(outputLayerNodeValues)
            
            # Hidden -> Hidden
            hiddenError = np.dot(outputDelta, outputWeights.T)
            hiddenDelta = hiddenError * self.sigmoid_der(hiddenLayerNodeValues[0])

            # Hidden -> Input
            inputError = np.dot(hiddenDelta, inputWeights.T)
            inputDelta = inputError * self.sigmoid_der(inputLayerNodeValues)
           
            # Calculate the adjustment to the weights
            inputAdj = np.dot(self.inData.T, inputDelta)
            hiddenAdj = np.dot(hiddenLayerNodeValues[0].T, hiddenDelta)
            outputAdj = np.dot(hiddenLayerNodeValues[-1].T, outputDelta)

            # Update the Weights
            inputWeights += inputAdj
            hiddenWeights[0] += hiddenAdj
            outputWeights += outputAdj

            # clays code
            # hiddenWeights = np.random.rand(len(self.inData[0]),4) 
            # outputWeights = np.random.rand(4,1)
            #Create values for the Hidden Layer
    #         hiddenWeightSum = np.dot(self.inData,hiddenWeights)
    #         hiddenFinalNodeValue = self.sigmoid(hiddenWeightSum)

    #         #Create values for the Output Layer
    #         outputWeightSum = np.dot(hiddenFinalNodeValue,outputWeights)
    #         outputFinalNodeValue = self.sigmoid(outputWeightSum)

    #         # ===[ Back Propigation ]===

    #         meanSquaredError = ((1/2) * (np.power((outputFinalNodeValue - self.inData), 2)))
    #         print("[" + str(epoch) + "]Error %: " + str(meanSquaredError.sum))

    #         #Update Output Weights
    #         outPredObserved = outputFinalNodeValue - self.outData #Difference between predicted data and actual data a.k.a the error
    #         outPred = self.sigmoid_der(outputWeightSum)

    #         outputBackProp = np.dot(hiddenFinalNodeValue.T,outPredObserved * outPred)

    #         #Update Hidden Weights
    #         hiddenVal = outPredObserved * outPred
            
    #         hiddenCost = np.dot(hiddenVal, outputWeights.T)
    #         hiddenBackProp = np.dot(self.outData.T,self.sigmoid_der(hiddenWeightSum) * hiddenCost)

    #         #Update Weights
    #         outputWeights -= learningRate * outputBackProp
    #         hiddenWeights -= learningRate * hiddenBackProp


    #     #Store
    #     self.trainedHidden = hiddenWeights
    #     self.trainedOutput = outputWeights 

    # def run(self, data):
    #     hiddenrun = self.sigmoid(np.dot(data,self.trainedHidden))
    #     output = self.sigmoid(np.dot(hiddenrun,self.trainedOutput))

    #     print(output)