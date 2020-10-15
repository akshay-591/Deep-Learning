import numpy as mat

"""numHiddenLayer = 1
numInputUnits = 3
numOutputUnits = 3
numHiddenUnits = 2

w1 = mat.arange(8).reshape(numHiddenUnits, numInputUnits + 1)
#w2 = mat.arange(6).reshape(numHiddenUnits[1], numHiddenUnits[0] + 1)
w3 = mat.arange(9).reshape(numOutputUnits, numHiddenUnits + 1)
print(" =========================== Before Reshaping ==================")
print("\n", w1)
#print("\n", w2)
print("\n", w3)

print("=============================== Testing ============================= ")
param = mat.r_[w1.flatten(), w3.flatten()]
weights = DeepNetsOptimization.reshapeWeights(param, numInputUnits, numOutputUnits, numHiddenLayer, numHiddenUnits)

print(len(weights))
print(weights['weights0'])
print(weights['weights1'])
#print(weights['weights2'].shape)"""


a = mat.arange(6).reshape(3,2)
b = mat.arange(9).reshape(3, 3)

para = mat.r_[a.flatten()]
para = mat.r_[para, b.flatten()]
print(para)