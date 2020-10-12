"""
This file will Initialize Weights randomly different for each unit/Neurons of different layers.
"""

import numpy as mat


def init(FirstLayerUnits, nextLayerUnits):
    """
    This method will initialize weights/parameters for a layer.

    :param FirstLayerUnits: input layer or feeding layer units
    :param nextLayerUnits: receiving layer or output layer units
    :return: Parameters of dimension nexLayerUnits*FirstLayerUnits+1 (including the bias weight)
    """
    e = 0.12  # for making values smaller

    parameters = mat.random.rand(nextLayerUnits, FirstLayerUnits + 1)

    # To make algorithm faster it is advisable to take small values of initial weights. To achieve that we will
    # multiply the weights with 2*e and subtract e It will give us small digits of random weights
    parameters = mat.multiply(parameters, 2 * e)
    parameters = mat.subtract(parameters, e)

    return parameters
