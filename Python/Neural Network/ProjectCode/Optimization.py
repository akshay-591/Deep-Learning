"""
This file Contains feedForward and BackPropagation method. Which is just like Cost/error and Gradient Descent but in
Neural Network they are called by different name because of their structure and working.
"""
import numpy as mat
from scipy import optimize
from ProjectCode import PredictionFunc, SigmoidDerivative


def Loss(param, X, Y, inputUnits, hiddenUnits, outputUnits, lamb):
    """
    This method will calculate the cost/error.

    :param param: flat array containing all weights (all layer).
    :param X: Input matrix.
    :param Y: output matrix.
    :param inputUnits: Input or initial layer units/Neurons mostly equal to total element of an image or number of
                       features.
    :param hiddenUnits: Hidden layer Units/Neurons.
    :param outputUnits: Output layer Units/Neuron which is equal to the number of classes.
    :param lamb: Regularization parameters.
    :return: Cost/error .
    """
    total_example = X.shape[0]

    # reshape the weight parameters
    weight1 = mat.reshape(param[0:((inputUnits + 1) * hiddenUnits)], (hiddenUnits, inputUnits + 1))
    weight2 = mat.reshape(param[((inputUnits + 1) * hiddenUnits):param.shape[0]], (outputUnits, hiddenUnits + 1))

    # call FeedForward Propagation and Calculate the output
    Hidden_output, output = FFProp(weight1, weight2, X)

    # Now Calculate the error between Original Output and the prediction

    """now we have classes to the equal number of outputs what we are going to do now will perform 1 vs all 
    Technique. In this Technique we are going to take 1 set and calculate the error by comparing it with one output 
    from one Neuron/unit for ex -  if we have 10 classes than the output will going have 10 Columns ( depends on how 
    data is arranged) Now we will take 1st set in which label 1 is going to be equal to 1 and rest will be 0 and then 
    we are going to subtract this set with Column 1 (or we can say output from neuron/unit 1 of output layer) only 
    from output and similarly set 2 with Column 2 set 3 with Column 3 and so on and at the end will add all the 
    errors that will give us the total error """

    # Here we are going to maximum likelihood Technique to calculate the errors

    loss0 = mat.zeros((outputUnits, 1))
    loss1 = mat.zeros((outputUnits, 1))

    for i in range(outputUnits):
        new_output = mat.where(Y == i+1, 1, 0)  # where Y == i replace with 1 and rest of the value to 0

        # Loss when when Y=1
        loss1[i, :] = mat.dot(-new_output.transpose(), mat.log(mat.c_[output[:, i]]))

        # loss when Y=0
        loss0[i, :] = mat.dot(mat.subtract(1, new_output).transpose(), mat.log(mat.subtract(1, mat.c_[output[:, i]])))

    # Total Avg loss

    loss_final = mat.multiply((1 / total_example), mat.sum(mat.subtract(loss1, loss0)))

    # Regularization regularize parameter = 1/2m * sum(theta(i)^2) from i=1 (i=0 or bias not included) to n where n
    # is number of features

    regularized_inputLayer = mat.multiply(lamb / (2 * total_example),
                                          mat.sum(mat.power(weight1[:, 1:weight1.shape[1]], 2)))

    regularized_outputLayer = mat.multiply(lamb / (2 * total_example),
                                           mat.sum(mat.power(weight2[:, 1:weight2.shape[1]], 2)))

    J = loss_final + regularized_inputLayer + regularized_outputLayer
    return J


def FFProp(weights1, weights2, X):
    """
    This Method is used to Execute FeedForward Propagation algorithm.

    :param weights1: layer one weights
    :param weights2: layer two weights.
    :param X: input matrix
    :return: Output from output layer
    """
    # add bias (ones column) in input or we can say output from initial layer
    X = mat.c_[mat.ones(X.shape[0]), X]

    # Calculate output of Hidden layer

    Hidden_output = PredictionFunc.predict(X, weights1, Sigmoid=True)

    # add bias (ones column) output from Hidden layer

    Hidden_output = mat.c_[mat.ones(Hidden_output.shape[0]), Hidden_output]

    # Calculate the prediction or output of output layer

    output = PredictionFunc.predict(Hidden_output, weights2, Sigmoid=True)

    return Hidden_output, output


def BackProp(param, X, Y, inputUnits, hiddenUnits, outputUnits, lamb):
    """
    This Method will execute BackPropagation algorithm and find the derivative w.r.t every parameters.

    :param param: flat array containing all weights (all layer).
    :param X: Input matrix.
    :param Y: output matrix.
    :param inputUnits: Input or initial layer units/Neurons mostly equal to total element of an image or number of
                       features.
    :param hiddenUnits: Hidden layer Units/Neurons.
    :param outputUnits: Output layer Units/Neuron which is equal to the number of classes.
    :param lamb: Regularization parameters.
    :return: derived function values with respect to every weight parameters.

    """
    total_example = X.shape[0]

    # reshape the weight parameters
    weight1 = mat.reshape(param[0:((inputUnits + 1) * hiddenUnits)], (hiddenUnits, inputUnits + 1))
    weight2 = mat.reshape(param[((inputUnits + 1) * hiddenUnits):param.shape[0]], (outputUnits, hiddenUnits + 1))

    # computes FastForward
    Hidden_output, output = FFProp(weight1, weight2, X)
    X = mat.c_[mat.ones(X.shape[0]), X]

    # ================================= Lets first Calculates f(J(theta) w.r.t Output layer weights ==================

    # calculates small delta values (errors) on each output unit.
    small_delta_output = mat.zeros((X.shape[0], outputUnits))
    for i in range(outputUnits):
        new_output = mat.where(Y == i+1, 1, 0)
        small_delta_output[:, i] = mat.subtract(mat.c_[output[:, i]], new_output).flatten()

    delta_output = mat.dot(small_delta_output.transpose(), Hidden_output)
    delta_output = mat.multiply(1 / total_example, delta_output)
    # regularization
    reg1 = mat.multiply(lamb / total_example, weight2[:, 1:weight2.shape[1]])

    temp_delta = mat.add(delta_output[:, 1:delta_output.shape[1]], reg1)
    reg_delta_output = mat.c_[delta_output[:, 0], temp_delta]

    # ======================================= Calculate for Hidden Layer ==============================================

    # calculate sigmoid derivative
    sigmoid_derivative = SigmoidDerivative.Sigmoid(Hidden_output, CalculateSig=False)

    # compute derivative
    small_delta_Hidden = mat.dot(small_delta_output, weight2)
    small_delta_Hidden = mat.multiply(small_delta_Hidden, sigmoid_derivative)
    small_delta_Hidden = mat.dot(small_delta_Hidden.transpose(), X)

    # Now small_delta_Hidden that means derivative of J(theta) w.r.t Hidden layer weights also contain extra bias weight
    # which we do not need so we will remove that weight.
    delta_Hidden = mat.multiply(1 / total_example, small_delta_Hidden[1:small_delta_Hidden.shape[0], :])

    # regularization
    reg2 = mat.multiply(lamb / total_example, weight1[:, 1:weight1.shape[1]])
    temp_delta = mat.add(delta_Hidden[:, 1:delta_Hidden.shape[1]], reg2)
    reg_delta_Hidden = mat.c_[delta_Hidden[:, 0], temp_delta]

    # zip all derivative in one flat array
    zip_derivatives = mat.r_[reg_delta_Hidden.flatten(), reg_delta_output.flatten()]

    return zip_derivatives


def optimize_grad(param, maxiter, args=()):
    """
    This methods Uses built in Conjugate Gradient method for optimization.

    :param param: are the parameters which user want to optimize.
    :param maxiter: maximum iteration.
    :param args: rest of the parameters for ex- Regularization parameter or Output etc.
    :return: Optimized parameters.
    """
    result = optimize.minimize(fun=Loss,
                               x0=param,
                               args=args,
                               method='CG',
                               jac=BackProp,
                               options={'maxiter': maxiter})
    return result
