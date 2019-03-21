import numpy as np

class Network:
    def __init__(self, inData, outData):
        self.inData = inData
        self.outData = outData
        self.trainedHidden = []
        self.trainedOutput = []

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def train(self,epochs):

        #Our hidden layer has 4 nodes, each input node need to connect to each hidden layer node
        hiddenWeights = np.random.rand(len(self.inData[0]),4) #4 is number of 
    
        #only 4 weights between hidden later and output
        outputWeights = np.random.rand(4,1)

        learnRate = 0.5

        for epoch in range(epochs):

            #=[ Feed Forward ]==============

            #Create values for the Hidden Layer
            hiddenWeightSum = np.dot(self.inData,hiddenWeights)
            hiddenFinalNodeValue = self.sigmoid(hiddenWeightSum)

            #Create values for the Output Layer
            outputWeightSum = np.dot(hiddenFinalNodeValue,outputWeights)
            outputFinalNodeValue = self.sigmoid(outputWeightSum)

            #=[ Back Propigation ]===========

            #TODO: Change this to a sum to reflect how the equation actually works with more that one hidden layer
            meanSquaredError = ((1/2) * (np.power((outputFinalNodeValue - self.inData),2)))
            print("[" + str(epoch) + "]Error %: " + str(meanSquaredError.sum))

            #Update Output Weights
            outPredObserved = outputFinalNodeValue - self.outData #Difference between predicted data and actual data a.k.a the error
            outPred = self.sigmoid_der(outputWeightSum)

            outputBackProp = np.dot(hiddenFinalNodeValue.T,outPredObserved * outPred)

            #Update Hidden Weights
            hiddenVal = outPredObserved * outPred
            hiddenCost = np.dot(hiddenVal, outputWeights.T)
            hiddenBackProp = np.dot(self.outData.T,self.sigmoid_der(hiddenWeightSum) * hiddenCost)

            #Update Weights
            outputWeights -= learnRate * outputBackProp
            hiddenWeights -= learnRate * hiddenBackProp


        #Store
        self.trainedHidden = hiddenWeights
        self.trainedOutput = outputWeights 

    def run(self, data):
        hiddenrun = self.sigmoid(np.dot(data,self.trainedHidden))
        output = self.sigmoid(np.dot(hiddenrun,self.trainedOutput))

        print(output)