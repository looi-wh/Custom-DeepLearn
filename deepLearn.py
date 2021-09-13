"""
deepLearn revision 5
new approach to machine learning


how it works:
 instead of using matrices and standard practices, this script attempts to solve machine learning by tons of arrays and maths
 basically: generate network --> backward propagation --> adjust network --> forward propagation --> compare output to expected
            repeat step 2 to step 4 for a million times and you will get a functioning model, at least in theory


functions list:
 ✅ = fully operational, 🥵 = core functions only, 🚨 = working on it, ⛔️ = stopped
 declareNodes 	(generate nodes network layout) [completed] ✅
 declareWeights (generate weights network layout) [completed] ✅
 dotProduct 	(calculate dot product) [completed] ✅
 forwardProp	(forward propagation) [completed] ✅
 backwardProp	(backward propagation) [completed] ✅
 viewGrowth		(generate a chart to show fitness level) 🚨
 selfLearnModel	(attempts to generate samples by itself) 🚨

learning algorithms:
 learnStandard	(supervised learning - attempts to provide the best weights that influence the closest result) [EXPERIMENTAL - still be tested] ✅
 learnSelf		()
 learnRandom 	(randomize bias and weights each time and save the best) 🚨
 learnLSTM		(provide memory cells for each nodes) 🚨

model usage:
 use-Model		(runs a simple forward propagation using the network model) 🚨

learnStandard:
 an awful bruteforce algorithm that requires samples and expected answers to adjust the network to

 the algorithm only adjusts weights instead of bias
 the weights are then adjusted to reduce the deviation between each output nodes with the expected answers
 then the weights will be kept between -1 and 1, and a sigmod function will be applied to weights exceeding limits
 there are also some rules to prevent weights from stalling

"""

import os
import sys
import math
import time
from random import seed
from random import random
from random import randint
from pathlib import Path # for managing file
from os import system, name # clearScreen
import pickle
import numpy as np # for sigmod function
import statistics

nodes = []
weights = []
biases = []
clearScreenOption = 1

# clearScreen functionality
def clearScreen(): # for screen clearing. can be disabled using clearScreenOption
	# version 3: optimized with better automation
	global clearScreenOption
	fault = 0
	if not clearScreenOption == 0:
		if name == "nt":
			fault = system("cls")
		else:
			fault = system("clear")
		if not fault == 0:
			clearScreenOption = 0 #disabling it just in case of any further use
	return fault

def declareNodes(inputNodes, hiddenNodes, hiddenLayers, outputNodes):
	# inputNodes: 	number of nodes in the input layer
	# hiddenNodes: 	number of nodes in each hidden layer
	# hiddenLayers:	number of layers in hidden layer
	# outputNodes:	number of nodes in output layer
	# this function will return an array with values
	# note: nodes and bias generation uses the same method, so bias will also use this function
	nodes = []
	inputNodesGEN = []
	for x in range(inputNodes):
		inputNodesGEN.append(random())
	nodes.append(inputNodesGEN)
	for x in range(hiddenLayers):
		hiddenNodesGEN = []
		for y in range(hiddenNodes):
			hiddenNodesGEN.append(random())
		nodes.append(hiddenNodesGEN)
	outputNodesGEN = []
	for x in range(outputNodes):
		outputNodesGEN.append(random())
	nodes.append(outputNodesGEN)
	return nodes

def declareWeights(nodes):
	# inputNodes: 	number of nodes in the input layer
	# hiddenNodes: 	number of nodes in each hidden layer
	# hiddenLayers:	number of layers in hidden layer
	# outputNodes:	number of nodes in output layer
	# weights have alot more values compared to nodes.
	# each nodes will have the same number of weights as the previous layer
	weights = []
	for layer in range(len(nodes)):
		if layer == 0:
			weightsGEN = []
			for x in range(len(nodes[layer])):
				weightsGEN.append([0])
			weights.append(weightsGEN)
		else:
			layerGEN = []
			for x in range(len(nodes[layer])):
				weightsGEN = []
				for y in nodes[layer - 1]:
					weightsGEN.append(random())
				layerGEN.append(weightsGEN)
			weights.append(layerGEN)
	return weights

def dotProduct(inputLayer, weightsOfNode, biasOfNode):
	# inputLayer:		the previous layer of target node
	# weightsOfNode:	the weights belonging to the target node
	# biasOfNode:		the bias belonging to target node
	return sum(inputLayer*weightsOfNode for inputLayer, weightsOfNode in zip(inputLayer, weightsOfNode)) + biasOfNode

def forwardProp(nodes, weights, bias):
	# forward propagate the network nodes
	# generate a value base on the previous layer
	for layer in range(len(nodes)):
		# 0 == input layer
		# input layer does not need any generation
		if layer > 0:
			for x in range(len(nodes[layer])):
				nodes[layer][x] = (dotProduct(nodes[layer - 1], weights[layer][x], bias[layer][x]))
	return nodes

def sigFunction(value): 
	# sigmoid function that converts all values in array to a value between 0 and 1
	# able to determine the input variable type and act accordingly
	if type(value) is list:
		outputx = []
		for x in range(0, len(value)):
			outputx.append(float(1/(1+np.exp(-value[x]))))
		return outputx
	elif type(value) is int or type(value) is float:
		return float(1/(1+np.exp(-value)))
	return 0

def tanhFunction(value):
	# tahn function is similar to sigmoid function, but convert values to a value between -1 and 1
	# 
	if type(value) is list:
		outputx = []
		for x in range(0, len(value)):
			outputx.append(float(math.tanh(value[x])))
		return outputx
	elif type(value) is int or type(value) is float:
		return float(math.tanh(value))
	return 0

def positiveFunction(value):
	# converts any negative numbers to positive only
	if type(value) is list:
		outputx = []
		for x in range(0, len(value)):
			if value[x] < 0:
				outputx.append(float(value[x]) * float(-1))
			else:
				outputx.append(float(value[x]))
		return outputx
	elif type(value) is int or type(value) is float:
		if value < 0:
			return float(float(value) * float(-1))
		else:
			return (float(value))
	return 0

def insertValues(nodes, inputx, layer):
	if len(inputx) == len(nodes[layer]):
		for x in range(len(inputx)):
			nodes[layer][x] = inputx[x]
		return nodes
	else:
		print("inputx length is not equal to nodes input layer")
		exit()

def backwardProp(nodes, weights, bias):
	# backward propagate the network
	# generate value base on the forward layer
	# this function basically inverse all the lists and send them to forwardProp
	nodesInverse = []
	for layer in range(len(nodes)):
		nodesInverse.append(nodes[len(nodes) - layer - 1])
	biasInverse = []
	for layer in range(len(bias)):
		biasInverse.append(bias[len(bias) - layer - 1])
	weightsInverseTEMP = []
	weightsInverse = []
	for layerx in range(len(nodes)-2, -1, -1):
		weights2GEN = []
		for x in range(len(nodes[layerx])):
			weights1GEN = []
			for node in range(len(nodes[layerx+1])):
				weights1GEN.append(weights[layerx+1][node][x])
			weightsInverse.append(weights1GEN)
		weights2GEN.append(weightsInverse)
	weightsInverseTEMP = weights2GEN[0]
	weightsINPUT = []
	for x in range(len(nodesInverse[0])):
		weightsINPUT.append(0)
	weightsInverse.append(weightsINPUT)
	for layer in range(1, len(nodesInverse), 1):
		weightsBetween = len(nodesInverse[layer])
		weightsGEN = []
		for x in range(weightsBetween):
			weightsGEN.append(weightsInverseTEMP[0])
			weightsInverseTEMP.pop(0)
		weightsInverse.append(weightsGEN)
	nodeResult = forwardProp(nodesInverse, weightsInverse, biasInverse)
	nodesInverse = []
	for layer in range(len(nodeResult)):
		nodesInverse.append(nodeResult[len(nodeResult) - layer - 1])
	return nodesInverse

def resultDeviation(output, expected):
	deviation = []
	if type(output) == list and type(expected) == list and len(output) == len(expected):
		for x in range(len(output)):
			difference = (expected[x] - output[x])
			deviation.append(difference*difference)
		return sum(deviation) / len(deviation)
	else:
		difference = (expected - output)
		return (difference*difference)

def generateFakeData(sets, inputNodes, outputNodes):
	# for testing purposes only
	# this function will generate random values with sample and test data
	# the model however will be terrible since
	samples = [[],[]]
	samples[0] = []
	samples[1] = []
	for x in range(sets):
		inputNodesGEN = []
		for y in range(inputNodes):
			inputNodesGEN.append(randint(-100, 100))
		samples[0].append(inputNodesGEN)
		outputNodesGEN = []
		for y in range(outputNodes):
			outputNodesGEN.append(randint(-100, 100))
		samples[1].append(outputNodesGEN)
	return samples

def learnStandardProposeWeights(nodes, weightsx, bias, samples, sample, precision, roundx, deviationAverage, timeTaken):
	# in order to simplify programming this monster, i have decided to import a certain part of the algorithmn into a function itself
	testSamples = samples[0]
	ansSamples = samples[1]
	output = []
	for layer in range(1, len(nodes)):
		for node in range(len(nodes[layer])):
			for o in range(len(nodes[-1])):
				output = forwardProp(nodes, weightsx,bias)
				deviation = resultDeviation(output[-1][o], ansSamples[sample][o])
				print("currently working on:", "round:", roundx+1,"sample:", sample+1, "layer:", layer, "node:",node, "deviation:", deviationAverage, "time taken:", int(timeTaken), "seconds       ", end="\r")
				for w in range(len(weightsx[layer][node])):
					if weightsx[layer][node][w] > 1 or weightsx[layer][node][w] < -1 : 		# <-- this feature acts as a controller to keep weights within a certain limit
						#weightsx[layer][node][w] = sigFunction(weightsx[layer][node][w]) 	# highly recommended, keeps value between 0 and 1
						weightsx[layer][node][w] = tanhFunction(weightsx[layer][node][w])	# keeps value between -1 and 1
						#weightsx[layer][node][w] = 0 										# last resort; makes use of the error correcting feature later in this script
					weightsBackup = weightsx[layer][node][w]
					if (weightsx[layer][node][w] < 0.0001 and weightsx[layer][node][w] > 0) or (weightsx[layer][node][w] == 0):
						purposedChange = 0.0001
					elif weightsx[layer][node][w] > -0.0001 and weightsx[layer][node][w] < 0:
						purposedChange = -0.0001
					else:
						purposedChange = (weightsx[layer][node][w]/100) * 5
					weightsx[layer][node][w] = weightsx[layer][node][w] + purposedChange
					output = forwardProp(nodes, weightsx, bias)
					purposedDeviation = resultDeviation(output[-1][o], ansSamples[sample][o])
					if purposedDeviation > deviation:
						weightsx[layer][node][w] = weightsBackup
						if (weightsx[layer][node][w] < 0.0001 and weightsx[layer][node][w] > 0) or (weightsx[layer][node][w] == 0):
							purposedChange = 0.0001
						elif weightsx[layer][node][w] > -0.0001 and weightsx[layer][node][w] < 0:
							purposedChange = -0.0001
						else:
							purposedChange = (weightsx[layer][node][w]/100) * 5
						weightsx[layer][node][w] = weightsx[layer][node][w] - purposedChange
						output = forwardProp(nodes, weightsx,bias)
						purposedDeviation = resultDeviation(output[-1][o], ansSamples[sample][o])
						if purposedDeviation > deviation:
							weightsx[layer][node][w] = weightsBackup
					elif purposedDeviation == deviation:
						weightsx[layer][node][w] = weightsBackup
	return weightsx


def learnStandard(nodes, weights, bias, samples, evalx, precision):
	# this function expects a complete network to be given and a proper dataset of test and answers
	# samples must be in the following format:
	#   [[list of inputs], [list of expected outputs]]
	#   samples[0] = inputs
	#   samples[1] = outputs
 	# 
	# both input and output must match the given network
	# function will also reserve 25% of samples for self testing
	#
	# this model learning method works by:
	# - tweaking weights and running forward prop during each tweaks
	# - reverts tweaks if deviation is higher than required
	# - tweaks are defined by percentage of weights itself.
	# - keeps weights values between -1 and 1
	# - if weights are neglated (near 0), function will submit a proposal of a small value to prevent stalling
	# - if weights is not between -1 and 1, a sigmod function will be applied to control the values within limits
	#
	#
	# recommended values:
	# evalx = 10 		# defines the number of rounds to run, higher values risks a higher chance of model memorising instead of learning
	# precision = 0.1 	# defines how strict the model should be, values should not be below 0 or above 1
	testSamples = samples[0]
	ansSamples = samples[1]
	timeTaken = 0
	output = []
	for roundx in range(evalx):
		purposedWeights = []
		for x in range(len(testSamples)): # filling up the purposedWeights list with initial data
			purposedWeights.append(0)
		startTime = time.time()
		for sample in range(len(testSamples)):
			nodes = insertValues(nodes, testSamples[sample], 0)
			deviationAverage = 1
			deviationRecord = []
			antiStall = 0
			antiStallFaultCounter = 0
			while deviationAverage > precision and antiStall == 0:
				try:
					purposedWeights[sample] = (learnStandardProposeWeights(nodes, weights, bias, samples, sample, precision, roundx, deviationAverage, timeTaken))
				except:
					return nodes, weights, bias
				deviationList = []
				output = forwardProp(nodes, weights, bias)
				for o in range(len(nodes[-1])):
					deviationList.append(resultDeviation(output[-1][o], ansSamples[sample][o]))
				deviationAverage = sum(deviationList) / len(deviationList)
				#
				# antiStall feature:
				# prevents the function from being "stuck" while trying to figure out the best weights value
				#
				deviationRecord.append(deviationAverage)
				if len(deviationRecord) > 20:
					antiStallLimit = sum(deviationRecord[-5:])/len(deviationRecord[-5:])
					deviationRecord.pop(0) # prevents 
					if sum(deviationRecord[5:])/5 < sum(deviationRecord[-5:])/5:
						averageDiff = 1
					else:
						averageDiff = 0
					if antiStallLimit <= deviationAverage or deviationRecord[-1] > deviationRecord[-5] or averageDiff == 1:
						antiStallFaultCounter += 1
						if antiStallFaultCounter >= 10: # the lower the value for antiStall, the more easily triggered it will get. if anti stall isnt working for you, lower this value instead.
							antiStall = 1
					else:
						antiStallFaultCounter = 0
		timeTaken = time.time() - startTime
		# WHAT IS "WEIGHTS FEATURE"
		# so learnStandard will actually "propose" weights for each specific sample that you have provided
		# then how can the algorithmn decide which weight is the "best" weight?
		# simple! we use another algorithmns to decide.
		# 
		#
		# average weights feature
		# a single weight = average of multiple proposed weights
		# advantages: easy to implement and works the fastest compared to the rest
		# disadvantages: not dynamic enough to adjust weights for best results
		#for layer in range(1, len(nodes)):
		#	for node in range(len(nodes[layer])):
		#		for w in range(len(weights[layer][node])):
		#			sumOfSingleWeightsAmongProposal = float(0)
		#			for proposal in range(len(purposedWeights)):
		#				sumOfSingleWeightsAmongProposal = sumOfSingleWeightsAmongProposal + float(purposedWeights[proposal][layer][node][w])
		#			weights[layer][node][w] = sumOfSingleWeightsAmongProposal/len(purposedWeights)
		#
		# best results weights feature
		# a single weight = compare all proposed weights and select the one which influences the lowest average deviation
		# advantages: after a certain period of time, the learning will actually speed up thanks to this algorthimn
		# disadvantages: requires more calculation as each possible weights will have to run a forward propagation for testing results
		#                so in the end, this might became an issue for larger sample sizes
		for z in range(len(nodes[-1])): # this feature actually doesnt need a scan thru component at the output layer, but I added it for precision purposes
			for layer in range(1, len(nodes)):
				for node in range(len(nodes[layer])):
					for w in range(len(weights[layer][node])):
						bestWeight = 0
						output = []
						output = forwardProp(nodes, weights, bias)
						weightsBackup = weights[layer][node][w]
						for o in range(len(nodes[-1])):
							deviationList.append(resultDeviation(output[-1][o], ansSamples[sample][o]))
						bestDeviation = sum(deviationList) / len(deviationList)
						firstDevation = bestDeviation
						for proposal in range(len(purposedWeights)):
							weights[layer][node][w] = purposedWeights[proposal][layer][node][w]
							output = forwardProp(nodes, weights, bias)
							for o in range(len(nodes[-1])):
								deviationList.append(resultDeviation(output[-1][o], ansSamples[sample][o]))
							purposedDeviation = sum(deviationList) / len(deviationList)
							if purposedDeviation < bestDeviation:
								bestWeight = weights[layer][node][w]
								bestDeviation = purposedDeviation
						# also proposes an average
						sumOfSingleWeightsAmongProposal = []
						for proposal in range(len(purposedWeights)):
							sumOfSingleWeightsAmongProposal.append(float(purposedWeights[proposal][layer][node][w]))
						purposedDeviation = sum(sumOfSingleWeightsAmongProposal) / len(sumOfSingleWeightsAmongProposal)
						if purposedDeviation < bestDeviation:
							bestWeight = weights[layer][node][w]
							bestDeviation = purposedDeviation
						if firstDevation > bestDeviation:
							weights[layer][node][w] = weightsBackup
						else:
							if not bestWeight == 0:
								weights[layer][node][w] = bestWeight
	return nodes, weights, bias


	
print("script started")
nodes = insertValues(declareNodes(3, 7, 5, 3), [0,0,0], 0)
weights = declareWeights(nodes)
bias = declareNodes(3, 7, 5, 3)
print("original")
print("network nodes values:")
for x in nodes:
	print(x)

print("")
print("")
print("total layers:", len(nodes))
print("")
print("")
samples = [[[0,20,0],[20,0,0],[0,0,20]],[[0,1,0],[1,0,0],[0,0,1]]]
nodes2, weights2, bias2 = learnStandard(insertValues(nodes, [0,0,0], 0) , weights, bias, samples, 10000, 0.1)
print("")
print("completed")
output = forwardProp(insertValues(nodes2, [0,0,20], 0), weights2, bias2)
#backwardProp(nodes , weights, bias)
print("network nodes values ([0,0,20]):")
print(output[-1])

print("")
output = forwardProp(insertValues(nodes2, [0,20,0], 0), weights2, bias2)
#backwardProp(nodes , weights, bias)
print("network nodes values ([0,20,0]):")
print(output[-1])

print("")
output = forwardProp(insertValues(nodes2, [20,0,0], 0), weights2, bias2)
#backwardProp(nodes , weights, bias)
print("network nodes values ([20,0,0]):")
print(output[-1])