"""
This is Neural Network Model for Hand Written Digit Main File.
"""

import numpy as mat
from scipy import io
from matplotlib import pyplot as plt
import random
from ProjectCode import Optimization, SigmoidDerivative, WeightInit, Prediction
from DebuggingTools import DebugOptimizationFun

#  load data
data = io.loadmat('../Data/trainSet.mat')

X_train = data['X']
Y_train = data['y']


input_layer_neurons = X_train.shape[1]  # input layer units
Hidden_layer_neuron = 25  # Hidden Layer units
numClasses = 10  # output layer units

# ============================== Visualization ====================================
# Visualizing 50 images randomly
fig, axes = plt.subplots(10, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    j = int(mat.round(random.random() * X_train.shape[0]))  # choose a index randomly

    im = mat.reshape(X_train[j, 0:X_train.shape[1]], (20, 20))  # reshape the row in 2D matrix

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
cost = Optimization.Loss(all_weights, X_train, Y_train, input_layer_neurons, Hidden_layer_neuron, numClasses, 0)
print("Cost Without Regularization = ", cost)

cost2 = Optimization.Loss(all_weights, X_train, Y_train, input_layer_neurons, Hidden_layer_neuron, numClasses, 1)
print("Cost With Regularization = ", cost2)
r = Optimization.BackProp(all_weights, X_train, Y_train, input_layer_neurons, Hidden_layer_neuron, numClasses, 0)

print("Checking derivative of sigmoid of values [-6 -0.5 0 0.8 10]")
test = mat.c_[mat.array([-6, -0.5, 0, 0.8, 10])]
result = SigmoidDerivative.Sigmoid(test.transpose(), CalculateSig=False)

print("Derived Value of sigmoid = ", result)

# ==================================== Apply Numerical vs Analytical approach ================================
print("\n\n========================== Debugging at Lambda = 0 =================== ")
DebugOptimizationFun.debug(0)
print("\n\n========================== Debugging at Lambda = 3 ======================= ")
DebugOptimizationFun.debug(3)

# ===================================== Training Neural Network =======================================
# ======================================= Initialize weights =========================================

"""
In neural Network we can not start with same or zero weights because it will give us the same output from all the 
units in hidden layer or second layer which is not what we want So it is necessary to initialize the weights randomly to
achieve non linear hypothesis.
"""

# for input layers
weights1 = WeightInit.init(input_layer_neurons, Hidden_layer_neuron)
# for Hidden layer
weights2 = WeightInit.init(Hidden_layer_neuron, numClasses)

all_param = mat.r_[layer1_Weights.flatten(), layer2_Weights.flatten()]

result = Optimization.optimize_grad(param=all_param,
                                    maxiter=100,
                                    args=(X_train, Y_train, input_layer_neurons, Hidden_layer_neuron, numClasses, 3))

learned_param = result.x
# reshape parameters

weight1 = mat.reshape(learned_param[0:((input_layer_neurons + 1) * Hidden_layer_neuron)],
                      (Hidden_layer_neuron, input_layer_neurons + 1))

weight2 = mat.reshape(learned_param[((input_layer_neurons + 1) * Hidden_layer_neuron):learned_param.shape[0]],
                      (numClasses, Hidden_layer_neuron + 1))
# ================================ Calculate Prediction and accuracy ==============================

prediction, accuracy = Prediction.predict(weight1, weight2, X_train, Y_train, Accuracy=True)
print("Model Accuracy is = ", accuracy)
