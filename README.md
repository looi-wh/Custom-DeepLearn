# Custom DeepLearn
Self made python script designed to simulate machine learning
And it is still being developed by me myself and I

## Disclaimer
I have absolute zero experience in machine learning and neither did I attended any classes on them
This is an attempt at machine learning by following different youtube videos and guides to create my own version of machine learning

DeepLearn is a simple script, that does not rely on tensorflow or any common machine learning tools

## Features

- Bunch of learning models ready to be called via a python function
- Highly customizable
- Uses the standard forward and backward propagation techniques
- Highly unstable, it is not production ready yet

## How it works:
Instead of using matrices and standard practices, this script attempts to solve machine learning by tons of arrays and maths
basically: 1) generate network --> 2) backward propagation --> 3) adjust network --> 4) forward propagation --> 5) compare output to expected
Repeat step 2 to step 5 for a million times and you will get a functioning model, at least in theory

## Functions list:
✅ = fully operational, 🥵 = core functions only, 🚨 = working on it, ⛔️ = stopped
- declareNodes 	(generate nodes network layout) [completed] ✅
- declareWeights (generate weights network layout) [completed] ✅
- dotProduct 	(calculate dot product) [completed] ✅
- forwardProp	(forward propagation) [completed] ✅
- backwardProp	(backward propagation) [completed] ✅
- viewGrowth		(generate a chart to show fitness level) 🚨
- selfLearnModel	(attempts to generate samples by itself) 🚨

## Learning algorithms:
- learnStandard	(supervised learning - attempts to provide the best weights that influence the closest result) [EXPERIMENTAL - still be tested] ✅
- learnSelf	    ()
- learnRandom 	(randomize bias and weights each time and save the best) 🚨
- learnLSTM		(provide memory cells for each nodes) 🚨

## Model usage:
- use-Model		(runs a simple forward propagation using the network model) 🚨

## learnStandard:
- an awful bruteforce algorithm that requires samples and expected answers to adjust the network to
- the algorithm only adjusts weights instead of bias
- the weights are then adjusted to reduce the deviation between each output nodes with the expected answers
- then the weights will be kept between -1 and 1, and a sigmod function will be applied to weights exceeding limits
- there are also some rules to prevent weights from stalling
