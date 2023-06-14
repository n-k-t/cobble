from __future__ import annotations
from buffer import Buffer

# Track the parents here for backprop?
class Operator():

    # Right now we are unable to access the parents from a given operator.
    # It doesn't appear to actually instantiate them. Maybe if I push 
    # return down a line, calculate the function, and then put it inside of 
    # a tensor afterwards.
    def __init__(self, *x: Tensor):
        self.parents = x

    def forward(self) -> 'None':
        raise NotImplementedError("This operation has not been implemented yet.")
    
    @classmethod
    def apply(function, *args, direction = 'forward', **kwargs) -> 'Tensor':
        if direction == 'forward':
            return Tensor(function.forward(*args), **kwargs, operator = function)
        else:
            raise NotImplementedError("This direction has not been implemented yet.")

# Must be here to prevent a circular import.
import operators

# May want to add in support to prevent mixed precision operations (i.e. fp16 and fp32).
# Either that or need to typecast them along the way.
class Tensor():

    def __init__(self, data, _children = (), operator = None):
        self.data = Buffer(data)
        self._previous = set(_children)
        self.operator = operator

    ######## UNARY OPERATORS ########

    ######## BINARY OPERATORS ########

    def add(self, x) -> 'Tensor':
        return operators.Addition.apply(self, x, _children = (self, x))
    
    def sub(self, x) -> 'Tensor':
        return operators.Subtraction.apply(self, x, _children = (self, x))
    
    def matmul(self, x) -> 'Tensor':
        return operators.MatrixMultiplication.apply(self, x, _children = (self, x))