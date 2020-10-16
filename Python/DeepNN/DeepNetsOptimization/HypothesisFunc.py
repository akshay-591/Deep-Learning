# this file contain sigmoid function

import numpy as mat


def predict(x, theta, Sigmoid=False):
    """
    This method calculate prediction for the Model

    :param x: input parameters
    :param theta: weight parameters
    :param Sigmoid: True means the prediction will be passed through a sigmoid function false just dot a product will
                    be calculated.
    :return: prediction

    """
    # calculating prediction
    if theta.shape[0] != x.shape[1] and theta.shape[1] == x.shape[1]:
        prediction = mat.dot(x, theta.transpose())
    else:
        prediction = mat.dot(x, theta)
    if Sigmoid:
        prediction = mat.divide(1, mat.add(1, mat.exp(-prediction)))

    return prediction
