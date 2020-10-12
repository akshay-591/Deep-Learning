"""
This file contains Method which will compute the prediction in form of original labels and return its prediction
and model accuracy if user wants.
"""
import numpy as mat
from ProjectCode import Optimization


def predict(Theta1, Theta2, X, Y, Accuracy=False):
    """
    This method calculates prediction and accuracy of the model.

    :param Accuracy: if True will be calculated for the model.
    :param Theta1: Input layer parameters.
    :param Theta2: Hidden Layer parameters.
    :param X: Input matrix.
    :param Y: Output matrix.
    :return: Prediction and Accuracy if True.
    """
    HiddenLayer, Output_layer = Optimization.FFProp(Theta1, Theta2, X)

    prediction = mat.argmax(Output_layer, axis=1) + 1  # add one because we have label from 1 to 10 but argument or max
    # index we will get will be from 0 to 9 so if original out put is 10.
    # we will get 9.

    if Accuracy:
        # calculate accuracy
        temp = mat.subtract(mat.c_[prediction], Y)
        accuracy = ((len(mat.where(temp == 0)[0])) / len(Y)) * 100
        return prediction, accuracy

    return prediction
