import numpy as np

# TODO define types here
class Array:
    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def shape(self):
        return self.tensor.shape

    def __str__(self):
        return "AD.array " + str(self.tensor)

def zeros(size, dtype=np.float32):
    tensor = np.zeros(size, dtype)
    return Array(tensor)

def ones(size, dtype=np.float32):
    tensor = np.ones(size, dtype)
    return Array(tensor)
