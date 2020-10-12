"""
This file Contain a method which will compute the derivative of Sigmoid function
"""

import numpy as mat


def Sigmoid(z, CalculateSig=True):
    """
    This methods Calculates the sigmoid derivative.

    :param z: input parameter
    :param CalculateSig: is True if Parameter needs to be passed through sigmoid function
                         first if Parameters is already passed then it should be false.
    :return: derived values of sigmoid
    """
    if CalculateSig:
        sig = 1 / (1 + mat.exp(-z))  # sigmoid
        ds = mat.multiply(sig, mat.subtract(1, sig))  # derivative of sigmoid function
    else:
        ds = mat.multiply(z, mat.subtract(1, z))
    return ds
