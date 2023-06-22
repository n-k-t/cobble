from __future__ import annotations
from buffer import Buffer

# Continue tracking the parents here for backprop?
class Operator():
    def __init__(self, *x: Tensor):
        self.parents = x

    def forward(self) -> 'None':
        raise NotImplementedError("This operation has not been implemented yet.")
    
    @classmethod
    def apply(function, *args, direction = 'forward', **kwargs) -> 'Tensor':
        if direction == 'forward':
            fxn = function(*args)
            return Tensor(fxn.forward(*args), **kwargs, operator = fxn)
        else:
            raise NotImplementedError("This direction has not been implemented yet.")

######## HELPER FUNCTIONS ########

# Get the strides associated with the shape of the tensor. Because arrays/tensors 
# are stored linearly in memory, the stride allows us to determine the number of 
# indices (memory addresses) are required to hop in order to index to a specific 
# element within the newly shaped tensor.
def get_stride(shape):
    stride = [1]

    for i in shape[::-1][:-1]:
        stride = [i * stride[0]] + stride

    return stride

# Must be here to prevent a circular import.
import operators

# May want to add in support to prevent mixed precision operations (i.e. fp16 and fp32).
# Either that or need to typecast them along the way.
class Tensor():
    def __init__(self, data, _children = (), operator = None):
        self.data = Buffer(data)
        self._previous = set(_children)
        self.operator = operator
        # May want to abstract this out to another class to make it easier to follow within.
        # These allow us to track and determine the shapes and strides of our tensors initially.
        #### I should add it so that these updated as the tensor gets manipulated.
        self.shape = self.data.buffer.shape
        self.stride = get_stride(self.shape)
        # These are here for when backpropagation gets implemented.
        # self.requires_gradient = requires_gradient
        # self.gradient = None

    ######## UNARY OPERATORS ########

    ######## BINARY OPERATORS ########

    def add(self, x) -> 'Tensor':
        return operators.Addition.apply(self, x, _children = (self, x))
    
    def sub(self, x) -> 'Tensor':
        return operators.Subtraction.apply(self, x, _children = (self, x))
    
    # May want to create the matrix in a different manner.
    #### On a forwards pass create it so that it must be transposed before 
    #### being multiplied.
    ######## https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/2
    def matmul(self, x) -> 'Tensor':
        return operators.MatrixMultiplication.apply(self, x, _children = (self, x))