from tensor import Operator
import numpy as np

# Add as many data attributes of the tensor as the user can handle.
class Addition(Operator):
    def forward(*args) -> 'ndarray':
        return np.add(*[x.data.buffer for x in args])
    
class Subtraction(Operator):
    def forward(*args) -> 'ndarray':
        return np.subtract(*[x.data.buffer for x in args])

class MatrixMultiplication(Operator):
    def forward(*args) -> 'ndarray':
        return np.matmul(*[x.data.buffer for x in args])
