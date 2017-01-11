import numpy as np

def read_cbin(filename):
    """
    loads .cbin files output by EvTAF
    """

    data = np.fromfile(filename,dtype=">d") # ">d" means big endian, double
    return data

def readrecf(filename):
    """
    reads .rec files output by EvTAF
    """
