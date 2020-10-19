"""
This is Neural Network Model for Hand Written Digit Main File.
"""

import numpy as mat
from scipy import io
from matplotlib import pyplot as plt
import random

from ProjectCode import Optimization, WeightInit, Prediction, Learn
from DebuggingTools import DebugOptimizationFun

#  load data
data = io.loadmat('../Data/trainSet.mat')

X = data['X']
Y = data['y']
print(Y)
# lets first choose the number of neuron we should use in Hidden layer using cross validation set
# we are going to take a difference between the units of 10.
Hidden_layer_neuron = 25
input_layer_neurons = X.shape[1]  # input layer units
numClasses = 10  # output layer units

# ============================== Visualization ====================================
# Visualizing 50 images randomly
fig, axes = plt.subplots(10, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    j = int(mat.round(random.random() * X.shape[0]))  # choose a index randomly

    im = mat.reshape(X[j, 0:X.shape[1]], (20, 20))  # reshape the row in 2D matrix

    ax.imshow(im.transpose(), 'gray')

plt.show()

# ===================================== Learning ====================================
# loading pre-initialized parameters
parameters = io.loadmat('../Data/testWeights.mat')

layer1_Weights = parameters['Theta1']
layer2_Weights = parameters['Theta2']

# zip them in single flat array

all_weights = mat.r_[layer1_Weights.flatten(), layer2_Weights.flatten()]

print(" ========================== Testing Methods ====================== ")
cost = Optimization.Loss(all_weights, X, Y, input_layer_neurons, Hidden_layer_neuron, numClasses, 0)
print("Cost Without Regularization = ", cost)

cost2 = Optimization.Loss(all_weights, X, Y, input_layer_neurons, Hidden_layer_neuron, numClasses, 1)
print("Cost With Regularization = ", cost2)
r = Optimization.BackProp(all_weights, X, Y, input_layer_neurons, Hidden_layer_neuron, numClasses, 0)

# ==================================== Apply Numerical vs Analytical approach ================================
print("\n\n========================== Debugging at Lambda = 0 =================== ")
DebugOptimizationFun.debug(0)
print("\n\n========================== Debugging at Lambda = 3 ======================= ")
DebugOptimizationFun.debug(3)

# ===================================== Training Neural Network =======================================

"""In neural Network we can not start with same or zero weights because it will give us the same output from all the 
units in hidden layer or second layer which is not what we want So it is necessary to initialize the weights randomly to
achieve non linear hypothesis.
"""
# for input layers
weights1 = WeightInit.init(input_layer_neurons, Hidden_layer_neuron)
# for Hidden layer
weights2 = WeightInit.init(Hidden_layer_neuron, numClasses)

all_param = mat.r_[weights1.flatten(), weights2.flatten()]

# starts Learning
learned_weight1, learned_weight2 = Learn.start(all_param, 50, X, Y, input_layer_neurons, Hidden_layer_neuron,
                                               numClasses, 1)
prediction, accuracy = Prediction.predict(learned_weight1, learned_weight2,
                                          X, Y, Accuracy=True)
print("Model Accuracy is = ", accuracy)

