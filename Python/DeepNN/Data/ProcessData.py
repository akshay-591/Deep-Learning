"""
In This file we will implement the code in which we will load the original data Shuffle that data and divide that data
in Train, test and cross validation sets 60% 20% 20% respectively and save that data in other .mat file so that
we do not have to do this Computation again and again when we work with our model.
"""

import numpy as mat
from scipy import io

# load original data
data = io.loadmat('OriginalMnistHandwritten.mat')

X = data['data'].transpose()  # extract input and keep the example in rows and pixels in columns
Y = data['label'].transpose()  # extract output and keep the labels in rows.
Y = mat.where(Y == 0, 10, Y)  # make the Value 10 where it is zero to work with our model.

"""
# Now we have 70000 examples we are going to divide them in three set.
# 1st will be a training set which will contain 60% of the data.
# 2nd will be cross- validation contain 50% of the left out data.
# and remaining 20% will be used as test set.
"""

# shuffle the data. Instead of shuffling the data we will going to shuffle the indexes and extract the data
# new matrix.

# get index
ind = mat.arange(X.shape[0])
# shuffle the index
mat.random.shuffle(ind)
# get the shuffle Data.

X_shuffled = X[ind, :]
Y_shuffled = Y[ind, :]

# some calculation so that it can work any number of example
numExample = int(X.shape[0])
numTrain = int(round(numExample * 0.6))  # 60% of total data set
numCrossV = int(round((numExample - numTrain) / 2)) + numTrain  # till here we going to extract data from original mat.

# Divide the data in training, Cross validation, Test set.

X_train = X_shuffled[0:numTrain, :]  # training set
Y_train = Y_shuffled[0:numTrain, :]

# CrossV
X_cv = X_shuffled[numTrain:numCrossV, :]
Y_cv = Y_shuffled[numTrain:numCrossV, :]

# Test set

X_test = X_shuffled[numCrossV:X_shuffled.shape[0], :]
Y_test = Y_shuffled[numCrossV:Y.shape[0], :]

# save the this whole data in .mat file so that we do not have do this calculation again and again this can save us
# a lot of computation time.

combined_data = {'X_train': X_train,
                 'Y_train': Y_train,
                 'X_cv': X_cv,
                 'Y_cv': Y_cv,
                 'X_test': X_test,
                 'Y_test': Y_test
                 }

io.savemat('ProcessedHandData.mat', combined_data)
