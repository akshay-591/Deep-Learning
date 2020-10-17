"""
This file contain method which will Compute the output using Feed forward propagation

"""
import numpy as mat
from DeepNetsOptimization import HypothesisFunc


def FFProp(X, weights):
    """
    This Method is used to Execute FeedForward Propagation algorithm.

    :param weights: Dictionary object containing reshaped weights.
    :param X: input matrix
    :return: Output from output layer
    """
    Layer_outputs = {}
    for i in range(len(weights) + 1):

        if i == 0:  # for initial layer
            # add bias (ones column) in input or we can say output from initial layer
            X = mat.c_[mat.ones(X.shape[0]), X]
            # Calculate output of Hidden layer0
            prediction = HypothesisFunc.predict(X, weights['weights' + str(i)], Sigmoid=True)
            Layer_outputs.update(
                {"Hidden_outputs" + str(i): mat.c_[mat.ones(prediction.shape[0]), prediction]})

        elif len(weights) - 1 > i:  # for Hidden layer
            # add bias
            po = Layer_outputs['Hidden_outputs' + str(i - 1)]  # previous output
            prediction = HypothesisFunc.predict(po, weights['weights' + str(i)], Sigmoid=True)
            Layer_outputs.update(
                {"Hidden_outputs" + str(i): mat.c_[mat.ones(prediction.shape[0]), prediction]})

        elif len(weights) - 1 == i:  # when we reached at Output later

            po = Layer_outputs['Hidden_outputs' + str(i - 1)]  # previous output
            prediction = HypothesisFunc.predict(po, weights['weights' + str(i)], Sigmoid=True)
            Layer_outputs.update(
                {"output_layer": prediction})

    return Layer_outputs
