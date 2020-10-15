"""
This is Main file for Deep Neural Network model in which we are going to use multiple number of Hidden layers and
neurons we are going to Build this model in such a way that it can work with any number of Hidden layer and neurons.

For This we have took Mnist data of Hand Written digit which have around 70000 samples of 28*28 images.
"""

import numpy as mat
from scipy import io
from DeepNetsOptimization import LossFunction, WeightInit, NNLearn
from DebuggingTools import DebugOptimizationFun
# load data
data = io.loadmat('../Data/ProcessedHandData.mat')

# Divide the data in training, Cross validation, Test set.

# training set.
X_train = data['X_train']
Y_train = data['Y_train']

# CrossV.
X_cv = data['X_cv']
Y_cv = data['Y_cv']

# Test set.
X_test = data['X_test']
Y_test = data['Y_test']

print("================================== Debugging at lambda = 0 ==================")
DebugOptimizationFun.debug(0)

print("================================== Debugging at lambda = 3 ==================")
DebugOptimizationFun.debug(3)

# lets first choose the number of neuron we should use in Hidden layer using cross validation set
# we are going to take a difference between the units of 10.
Hidden_layer = 1
Hidden_layer_neuron = 25
input_layer_neurons = X_train.shape[1]  # input layer units
numClasses = 10  # output layer units

# ===================================== Learning ====================================
all_param = WeightInit.init(input_layer_neurons, numClasses, Hidden_layer, Hidden_layer_neuron)

cost = LossFunction.Loss(all_param, X_train, Y_train, input_layer_neurons, numClasses, Hidden_layer,
                         Hidden_layer_neuron, 0)

model = NNLearn.Learn(Input=X_train,
                      Output=Y_train,
                      InputLayerUnits=input_layer_neurons,
                      OutputLayerUnits=numClasses,
                      numHiddenLayers=Hidden_layer,
                      HiddenLayerUnits=Hidden_layer_neuron,
                      lamb=1)
print("Model Accuracy is = ", model.accuracy)
