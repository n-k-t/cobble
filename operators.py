from tensor import Operator
import numpy as np

######## BASE OPERATORS ########

# This is a dict for now allowing us to store functions to call in the 
# future, however should still consider whether or not this is the right 
# approach.

# Dropped the dict and made a class to allow us to create an enum.
# Using a base class instead of an enum is faster on lookup.
# None is smaller than lambda: None

class BaseOperators():
    ADD = None
    SUBTRACT = None
    MULTIPLY = None
    POWER = None


# Add as many data attributes of the tensor as the user can handle.
class Addition(Operator):
    def forward(self, *args) -> 'ndarray':
        return np.add(*[x.data.buffer for x in args])
    
class Subtraction(Operator):
    def forward(self, *args) -> 'ndarray':
        return np.subtract(*[x.data.buffer for x in args])

class MatrixMultiplication(Operator):
    def forward(self, *args) -> 'ndarray':
        return np.matmul(*[x.data.buffer for x in args])
