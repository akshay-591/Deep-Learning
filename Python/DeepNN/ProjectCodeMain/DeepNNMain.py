"""
This is Main file for Deep Neural Network model in which we are going to use multiple number of Hidden layers and
neurons we are going to Build this model in such a way that it can work with any number of Hidden layer and neurons.

For This we have took Mnist data of Hand Written digit which have around 70000 samples of 28*28 images.
"""

import numpy as mat
from scipy import io
from DeepNetsOptimization import NNLearn, Prediction
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

print("================================== Debugging The Implementation of Methods ==================")
DebugOptimizationFun.debug(HiddenActivation="ReLu",
                           OutputActivation="Sigmoid",
                           InputUnits=6,
                           OutputUnits=4,
                           numHiddenLayer=2,
                           numExample=3,
                           lamb=0)

print("==================================== Learning =================================================")
# Scale the pixel between 0 and 1
X_scale_train = mat.divide(X_train, 255)
X_scale_cv = mat.divide(X_cv, 255)
X_scale_test = mat.divide(X_test, 255)


Hidden_layer = 1
Hidden_layer_neuron = 50
input_layer_neurons = X_train.shape[1]  # input layer units
numClasses = 10  # output layer units
model = NNLearn.Learn(Input=X_scale_train,
                      Output=Y_train,
                      AutoParameters=True,
                      maxIter=50,
                      InputLayerUnits=input_layer_neurons,
                      OutputLayerUnits=numClasses,
                      numHiddenLayers=Hidden_layer,
                      HiddenLayerUnits=Hidden_layer_neuron,
                      HiddenActivation="ReLu",
                      OutputActivation="Sigmoid",
                      lamb=1)

model = NNLearn.startTraining(model)
print("Accuracy on Training set = ", model.accuracy)
# Accuracy On cross Validation set
prediction, accuracy = Prediction.predict(model, X_scale_cv, Y_cv, Accuracy=True)
print("Accuracy on Cross Validation set is = ", accuracy)
prediction_t, accuracy_t = Prediction.predict(model, X_scale_test, Y_test, Accuracy=True)
print("Accuracy on Test set is = ", accuracy_t)
