"""
This File Contain method which will compute the derivative using BackPropagation.
"""
import numpy as mat

from DeepNetsOptimization import SigmoidDerivative
from DeepNetsOptimization.ForwardPropagation import FFProp
from DeepNetsOptimization.Reshape import reshapeWeights


def BackProp(param, X, Y, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits, lamb):
    """
    This Method will execute BackPropagation algorithm and find the derivative w.r.t every parameters.

    :param param: flat array containing all weights (all layer).
    :param X: Input matrix.
    :param Y: output matrix.
    :param inputUnits: Input or initial layer units/Neurons mostly equal to total element of an image or number of
                       features.
    :param numHiddenLayer: Number of Hidden Layer.
    :param numHiddenUnits: Hidden layer Units/Neurons.
    :param outputUnits: Output layer Units/Neuron which is equal to the number of classes.
    :param lamb: Regularization parameters.
    :return: derived function values with respect to every weight parameters.

    """
    total_example = X.shape[0]
    small_delta_previous = {}
    weights = reshapeWeights(param, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits)

    # call FeedForward Propagation and Calculate the outputs
    Layer_outputs = FFProp(X, weights)

    # add bias to X matrix
    X = mat.c_[mat.ones(X.shape[0]), X]

    # initialize parameters matrix which need to return
    derivatives = mat.zeros(param.shape[0])

    numTrace = 0  # trace variable
    for i in reversed(range(len(weights))):

        if i == numHiddenLayer:
            """
            The loop will enter here when measuring d(j) w.r.t to then weights which are in between last Hidden layer 
            and the Output Layer.
            """
            output = Layer_outputs['output_layer']
            # calculates small delta values (errors) on each output unit.
            small_delta_output = mat.zeros((X.shape[0], outputUnits))
            for j in range(outputUnits):
                new_output = mat.where(Y == j + 1, 1, 0)
                small_delta_output[:, j] = mat.subtract(mat.c_[output[:, j]], new_output).flatten()

            # Store the delta values which will be useful in next layer calculation (we are In Backward Direction)
            small_delta_previous.update({"smalldelta": small_delta_output})

            # Compute the Capital delta means multiply to the Input Coming in the Layer Which is Output From
            # Last Hidden Layer.

            delta_output = mat.dot(small_delta_output.transpose(), Layer_outputs['Hidden_outputs' + str(i - 1)])
            delta_output = mat.multiply(1 / total_example, delta_output)

            # regularization
            reg1 = mat.multiply(lamb / total_example,
                                weights['weights' + str(i)][:, 1:weights['weights' + str(i)].shape[1]])

            temp_delta = mat.add(delta_output[:, 1:delta_output.shape[1]], reg1)
            reg_delta_output = mat.c_[delta_output[:, 0], temp_delta]

            # appending the output in derivative in last
            TotalElements = mat.prod(reg_delta_output.shape)

            numTrace = param.shape[0] - TotalElements  # This will help us from which index we should store the values
            # so that These derivative remain in the last of array.

            derivatives[numTrace:param.shape[0]] = reg_delta_output.flatten()

        if 0 < i < numHiddenLayer:
            """ 
            will be enter here only if The Hidden Layer are more than one,means when calculating d(j) w.r.t to the 
            weights between Hidden Layers for ex- if we have two Hidden Layer so loop will enter here only once.
            """
            derivatives, numTrace = CommonCodeBackProp(i=i,
                                                       OutPut=Layer_outputs['Hidden_outputs' + str(i)],
                                                       Input=Layer_outputs['Hidden_outputs' + str(i - 1)],
                                                       PreviousDelta=small_delta_previous,
                                                       weights=weights,
                                                       numTrace=numTrace,
                                                       derivatives=derivatives,
                                                       totalExample=total_example,
                                                       lamb=lamb
                                                       )

        if i == 0:
            """
             when reached at Input Layer, means measuring d(j) w.r.t weights between Input layer and 1st Hidden layer.
             """
            derivatives, numTrace = CommonCodeBackProp(i=i,
                                                       OutPut=Layer_outputs['Hidden_outputs' + str(i)],
                                                       Input=X,
                                                       PreviousDelta=small_delta_previous,
                                                       weights=weights,
                                                       numTrace=numTrace,
                                                       derivatives=derivatives,
                                                       totalExample=total_example,
                                                       lamb=lamb
                                                       )

    return derivatives


def CommonCodeBackProp(i, OutPut, Input, PreviousDelta, weights, numTrace, derivatives, totalExample, lamb):
    """
    This Method contain common code for Computing the BackPropagation for Layer other than output.

    :param i: Loop Number
    :param OutPut: The Output Computed by the layer at which the current loop is for computing Sigmoid derivative of
                   Output. Current Layer is Considered which is on the Right side of The Weights.

    :param Input: The Input which is Coming In the layer at Which the current loop is.
                   Current Layer is Considered which is on the Left side of The Weights.

    :param PreviousDelta: small delta Computed till Previous layer.
    :param weights: Dictionary  Object Containing all reshaped weights.
    :param numTrace: Trace variable.
    :param derivatives: derivative list in which all the derived valued are stored.
    :param totalExample: Total number of example.
    :param lamb: Regularization Parameter.
    :return: Updated numTrace and derivatives list.
    """
    # calculate sigmoid derivative Which is going Out From Current Layer
    sigmoid_derivative = SigmoidDerivative.Sigmoid(OutPut, CalculateSig=False)

    # Multiply the Previous Store delta with the Previous weights.
    small_delta_Hidden = mat.dot(PreviousDelta['smalldelta'], weights['weights' + str(i + 1)])

    small_delta_Hidden = mat.multiply(small_delta_Hidden, sigmoid_derivative)

    # Now small_delta_Hidden that means derivative of J(theta) w.r.t Hidden layer weights also contain extra
    # bias weight which we do not need so we will remove that weight.

    small_delta_Hidden = small_delta_Hidden[:, 1:small_delta_Hidden.shape[1]]

    # update previous delta value
    PreviousDelta.update({"smalldelta": small_delta_Hidden})

    # Compute the Capital delta means multiply to the Input.
    delta_Hidden = mat.dot(small_delta_Hidden.transpose(), Input)

    delta_Hidden = mat.multiply(1 / totalExample, delta_Hidden)

    # regularization
    # extract weights for regularization from Dictionary since we are not going to regularize the bias weights

    parm = weights['weights' + str(i)]
    reg2 = mat.multiply(lamb / totalExample, parm[:, 1:parm.shape[1]])

    temp_delta = mat.add(delta_Hidden[:, 1:delta_Hidden.shape[1]], reg2)
    reg_delta_Hidden = mat.c_[delta_Hidden[:, 0], temp_delta]

    # appending the output in derivative
    TotalElements = mat.prod(reg_delta_Hidden.shape)
    numTrace = numTrace - TotalElements
    derivatives[numTrace:TotalElements + numTrace] = reg_delta_Hidden.flatten()

    return derivatives, numTrace
