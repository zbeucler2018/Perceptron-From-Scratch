


import random, math

"""
Simple perceptron with backpropagation from scratch by Zack Beucler

"""



class Perceptron:

	def __init__(self, numInputs, learningRate, epochs, isVerbose=False ):
		self.numInputs = numInputs
		self.weights = []
		self.threshold = -1
		self.thresholdWeight = None
		self.learningRate = learningRate
		self.epochs = epochs
		self.isVerbose = isVerbose

	def setWeights(self, value):
		self.weights = value

	def getWeights(self):
		return self.weights

	def setThresholdWeight(self, value):
		self.thresholdWeight = value

	def getThresholdWeight(self):
		return self.thresholdWeight

	def sigmoid(self, value):
		return 1 / (1 + math.exp(-1 * value))

	def activation(self, inputs, weights, threshold, thresholdWeight):
		dotProduct = sum([i*j for (i, j) in zip(inputs, weights)])  # get the dot product of the weights and inputs
		total = dotProduct + (threshold * thresholdWeight) # bias the dot product
		return self.sigmoid(total)

	def train(self, trainingData):
		"""
		Trains the perceptron using back propagation.
		Data needs to be in this format [[inputs] output]
			EX: [[1,1,1,1,0], 0]
		"""
		if self.weights == []: 
			self.setWeights([random.uniform(-0.5,0.5) for x in range(self.numInputs)]) # set the random weights
		if self.thresholdWeight is None:
			self.setThresholdWeight(random.uniform(-0.5, 0.5)) # set random threshold weight
		for epoch in range(self.epochs):
			if self.isVerbose: print(f"Epoch {epoch} out of {self.epochs}")
			for indx,sample in enumerate(trainingData):
				if self.isVerbose: print(f"\tSample {indx} out of {len(trainingData)}")
				inputs = sample[0] # extract the inputs
				desiredOutput = sample[1] # get the desired output
				perceptronOutput = self.activation(inputs, self.weights, self.threshold, self.thresholdWeight) # get the perceptron's output
				error = desiredOutput - perceptronOutput # calculate error
				### threshold weight correction calculation ###
				deltaThresholdWeight = self.learningRate * self.threshold * error
				newThresholdWeight = self.thresholdWeight + deltaThresholdWeight
				self.setThresholdWeight(newThresholdWeight) # update the threshold weight
				### weights correction calculations ###
				newWeights = []
				for w_indx, currentWeight in enumerate(self.weights):
					currentInput = inputs[w_indx]
					deltaWeight = self.learningRate * currentInput * error
					newWeight = currentWeight + deltaWeight
					newWeights.append(newWeight)

				self.setWeights(newWeights) # update the global weights



	def test(self, testData, closeness=0.1):
		"""
		test the trained.
		Data must be in format [[input], output]
		"""
		totalCorrect = 0
		for indx,sample in enumerate(testData):
			inputs = sample[0] # extract the inputs
			desiredOutput = sample[1] # get the desired output
			perceptronOutput = self.activation(inputs, self.weights, self.threshold, self.thresholdWeight) # get the perceptron's output
			if abs(perceptronOutput-desiredOutput) <= closeness:
				totalCorrect += 1
				if self.isVerbose: print("Actual:", perceptronOutput, "\tDesired:", desiredOutput, "correct")
			else:
				if self.isVerbose: print("Actual:", perceptronOutput, "\tDesired:", desiredOutput)


		print("Correct guesses:", totalCorrect, "out of", len(testData))


	def predict(self, inputs):
		"""
		Use the perceptron to predict on inputs.
		Data must be in an array
		"""
		return self.activation(inputs, self.weights, self.threshold, self.thresholdWeight)









			




def main():
	trainingSample = [
		[[0,0,0,0,0], 0],
		[[0,0,0,0,1], 0],
		[[0,0,0,1,0], 0],
		[[0,0,1,0,0], 0],
		[[0,1,0,0,0], 0],
		[[1,0,0,0,0], 0],
		[[0,0,0,1,1], 0],
		[[0,0,1,0,1], 0],
		[[0,1,0,0,1], 0],
		[[1,0,0,0,1], 0],
		[[0,0,1,1,0], 0],
		[[0,1,0,1,0], 0],
		[[1,0,0,1,0], 0],
		[[0,1,1,0,0], 0],
		[[1,0,1,0,0], 0],
		[[1,1,0,0,0], 0],
		[[0,0,1,1,1], 0],
		[[0,1,0,1,1], 0],
		[[1,0,0,1,1], 0],
		[[0,1,1,0,1], 0],
		[[1,0,1,0,1], 0],
		[[1,1,0,0,1], 0],
		[[0,1,1,1,0], 0],
		[[1,0,1,1,0], 0],
		[[1,1,0,1,0], 0],
		[[1,1,1,0,0], 0],
		[[0,1,1,1,1], 1],
		[[1,0,1,1,1], 1],
		[[1,1,0,1,1], 1],
		[[1,1,1,0,1], 1], 
		[[1,1,1,1,0], 1],
		[[1,1,1,1,1], 1]
	]

	p = Perceptron(5, 0.01, 50000, )
	p.train(trainingSample)
	#i = p.getWeights()
	p.test(trainingSample)

	print(p.predict([1,1,1,1,1]), "should be close to 1")



main()
