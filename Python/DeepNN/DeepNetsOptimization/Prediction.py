"""
This file contains Method which will compute the prediction in form of original labels and return its prediction
and model accuracy if user wants.
"""
import numpy as mat
from DeepNetsOptimization import LossFunction, Reshape


def predict(model, X, Accuracy=False):
    """
    This method calculates prediction and accuracy of the model.

    :param X: Input
    :param model: Container Containing all the Information. Like Number of Neurons in Layers and number of Hidden Layers
                  and Learned weights.
    :param Accuracy: if True will be calculated for the model.
    :return: Prediction and Accuracy if True.
    """

    reshped_weights = Reshape.reshapeWeights(model.learnedWeights, model.InputLayerUnits,
                                             model.OutputLayerUnits, model.numHiddenLayers,
                                             model.HiddenLayerUnits)

    Layers_output = LossFunction.FFProp(X, reshped_weights)
    Output_layer = Layers_output['output_layer']
    prediction = mat.argmax(Output_layer, axis=1) + 1  # add one because we have label from 1 to 10 but argument or max
    # index we will get will be from 0 to 9 so if original output is 10 we will get 9 That will give the wrong
    # Prediction and 0 accuracy.

    if Accuracy:
        # calculate accuracy
        error = mat.subtract(mat.c_[prediction], model.Y)  # subtract the Prediction from Original Output
        accuracy = ((len(mat.where(error == 0)[0])) / len(model.Y)) * 100  # Calculate where error is 0.
        return prediction, accuracy

    return prediction
