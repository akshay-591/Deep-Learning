"""
This file will Initialize Weights randomly different for each unit/Neurons of different layers.
"""

import numpy as mat


def init(InputUnits, OutputUnits, numHiddenLayer=1, HiddenUnits=None):
    """
    This Method will initialize the Weights randomly.

    :param Activation:
    :param InputUnits: Number of Units in InputLayer
    :param OutputUnits: Number of Classes
    :param numHiddenLayer: Number of Hidden Layer
    :param HiddenUnits: Number of HiddenUnits. If Hidden Layer is more than one append the units in list respectively
    :return: flat array containing all weights
    """
    global HiddenUnit
    all_weights = []
    if HiddenUnits is None:
        HiddenUnit = []
    elif isinstance(HiddenUnits, int):
        HiddenUnit = [HiddenUnits]
    else:
        HiddenUnit = HiddenUnits

    # for InputLayer

    parameters = Compute2(InputUnits, HiddenUnit[0])
    allWeights = mat.r_[parameters.flatten()]

    if numHiddenLayer > 1:
        for i in range(numHiddenLayer):
            if i < numHiddenLayer - 1:
                parameters = Compute2(HiddenUnit[i], HiddenUnit[i + 1])
                allWeights = mat.r_[allWeights, parameters.flatten()]
            else:
                parameters = Compute(HiddenUnit[i], OutputUnits)
                allWeights = mat.r_[allWeights, parameters.flatten()]

    else:
        # for output layer
        parameters = Compute(HiddenUnit[0], OutputUnits)
        allWeights = mat.r_[allWeights, parameters.flatten()]

    return allWeights


def Compute(FirstLayerUnits, nextLayerUnits):
    """
    This Method will Compute the parameters for and return it in dimension of nextLayerUnits*FirstLayerUnits+1

    :param FirstLayerUnits: The units in Current Layer
    :param nextLayerUnits: The units in Next Layer
    :return: parameters matrix of dimension of nextLayerUnits*FirstLayerUnits+1
    """

    e = 0.12  # for making values smaller
    parameters = mat.random.randn(nextLayerUnits, FirstLayerUnits + 1)
    # To make algorithm faster it is advisable to take small values of initial weights. To achieve that we will
    # multiply the weights with 2*e and subtract e It will give us small digits of random weights
    parameters = mat.multiply(parameters, 2 * e)
    parameters = mat.subtract(parameters, e)

    return parameters


def Compute2(FirstLayerUnits, nextLayerUnits):
    """
    This Method will Compute the parameters for and return it in dimension of nextLayerUnits*FirstLayerUnits+1

    :param FirstLayerUnits: The units in Current Layer
    :param nextLayerUnits: The units in Next Layer
    :return: parameters matrix of dimension of nextLayerUnits*FirstLayerUnits+1
    """

    n = nextLayerUnits
    parameters = mat.multiply(mat.random.randn(nextLayerUnits, FirstLayerUnits + 1), mat.sqrt(1/n))
    # To make algorithm faster it is advisable to take small values of initial weights. To achieve that we will
    # multiply the weights with 2*e and subtract e It will give us small digits of random weights

    return parameters
