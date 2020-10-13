"""
The file contains method which will starts the learning process of Neural Network.
"""
import numpy as mat

from ProjectCode import Optimization


def start(param, maxIter, X, Y, input_layer_neurons, Hidden_layer_neuron, numClasses, lamb):
    """
    This method will initialize the learning process.

    :param param: parameters
    :param maxIter: max Iteration
    :param X: input
    :param Y: output
    :param input_layer_neurons: Number of input layer neurons
    :param Hidden_layer_neuron: Number of Output Layer neurons
    :param numClasses: Number of classes
    :param lamb: Regularization parameter
    :return: weights 1 and Weights 2.
    """
    result = Optimization.optimize_grad(param=param,
                                        maxiter=maxIter,
                                        args=(X, Y, input_layer_neurons,
                                              Hidden_layer_neuron, numClasses, lamb))
    learnedWeights = result.x
    # reshape parameters

    learned_weight1 = mat.reshape(learnedWeights[0:((input_layer_neurons + 1) * Hidden_layer_neuron)],
                                  (Hidden_layer_neuron, input_layer_neurons + 1))

    learned_weight2 = mat.reshape(
        learnedWeights[((input_layer_neurons + 1) * Hidden_layer_neuron):learnedWeights.shape[0]],
        (numClasses, Hidden_layer_neuron + 1))

    return learned_weight1, learned_weight2
