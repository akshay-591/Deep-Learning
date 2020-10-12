"""
This file Contains Method which will debug the Optimization Functions Using Numerical Vs Analytical approach
"""

import numpy as mat
from DebuggingTools import WeightsDebug, TestNumericalGradient
from ProjectCode import Optimization


def debug(lamb=0):
    """
    This method will generate some random data and and Calculate Gradients or Derivative of Function Numerically and
    Analytical both.

    :param lamb: Regularization parameter By default it is 0.
    :return: Numerical Gradients , Analytical Gradients and difference/error.
    """

    # define some parameters for temp data
    inLayerSize = 6
    HiddenLayerSize = 5
    numClasses = 4
    examples = 5

    # Generate some Weight parameters
    layer1_param = WeightsDebug.generate(HiddenLayerSize, inLayerSize)
    layer2_param = WeightsDebug.generate(numClasses, HiddenLayerSize)

    # Generate some input and output data
    X = WeightsDebug.generate(examples, inLayerSize - 1)
    Y = 1 + mat.reshape(mat.arange(start=1, stop=examples + 1, step=+1), (examples, 1)) % numClasses

    # zip parameters
    param = mat.r_[layer1_param.flatten(), layer2_param.flatten()]

    # Calculates numerical gradients

    numerical_values = TestNumericalGradient.NumGrad(function=Optimization.Loss,
                                                     theta=param,
                                                     parameters=(X, Y, inLayerSize, HiddenLayerSize, numClasses, lamb))

    # Calculates Analytical gradients

    Analytical_values = Optimization.BackProp(param, X, Y, inLayerSize, HiddenLayerSize, numClasses, lamb)

    # calculate difference
    mat_a = mat.subtract(numerical_values, Analytical_values)
    mat_b = mat.add(numerical_values, Analytical_values)
    # calculate norm
    diff = mat.linalg.norm(mat_a) / mat.linalg.norm(mat_b)

    # print the values
    print("\nNumerical Calculated Gradients = \n", numerical_values)
    print("\nAnalytical Calculated Gradients = \n", Analytical_values)
    print("\ndifference = ", diff)
    print("\nif the both the Values are almost same and Difference is less than 1e-9 than test is Successful")

    return numerical_values, Analytical_values, diff