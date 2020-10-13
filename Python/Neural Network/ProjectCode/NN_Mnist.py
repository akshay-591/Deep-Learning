""""
In This file we will do the actual Training on Different and bigger data provides at Mnist website.
"""
import numpy as mat
from scipy import io

"""# ===================================== Calibrating ==============================================================
print(" =========================================== Hidden Layer Unit Test begin ==============")
units_test = [50, 53, 55, 58, 60, 63, 65, 67, 69, 70]
accuracy_total = mat.zeros((len(units_test), 1))
j = 0
for i in units_test:
    Hidden_layer_neuron = i  # Hidden Layer units
    # for input layers
    weights1 = WeightInit.init(input_layer_neurons, Hidden_layer_neuron)

    # for Hidden layer
    weights2 = WeightInit.init(Hidden_layer_neuron, numClasses)

    accuracy_total[j, :] = accuracy
    j += 1
    print("Model Accuracy at number of units", i, " is ", accuracy)
    print("\n")

# number of Hidden layer unit is
index_ofmax = mat.argmax(accuracy_total).item()
print(index_ofmax)
Hidden_layer_neuron = units_test[index_ofmax]
print(Hidden_layer_neuron)"""
