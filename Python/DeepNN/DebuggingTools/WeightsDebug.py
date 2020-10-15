"""
This File Contains Method for Generating some Data which is useful for Debugging Model.
"""

import numpy as mat


def generate(inUnits, OutUnits):
    """
    This method will Generate Data for Debugging.

    :param inUnits: Layer unit in which data is going.
    :param OutUnits: Layer unit from which data is going.
    :return: parameters of dimension inUnits*OutUnits+1.
    """
    array = mat.arange(start=1, stop=(inUnits * (OutUnits + 1))+1, step=1)
    debugWeights = mat.divide(mat.reshape(mat.sin(array), (inUnits, (OutUnits+1))),10)

    return debugWeights



