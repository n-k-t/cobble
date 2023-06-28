from __future__ import annotations
from lazy_tensor import LazyOperator, LazyTensor

# Continue tracking the parents here for backprop?
class Operator():
    def __init__(self, *x: Tensor):
        self.parents = set(x)

    def forward(self) -> 'None':
        raise NotImplementedError("This operation has not been implemented yet.")
    
    @classmethod
    def apply(function, *args, direction = 'forward', **kwargs) -> 'Tensor':
        if direction == 'forward':
            fxn = function(*args)
            return Tensor(fxn.forward(*args), **kwargs, operator = fxn)
        else:
            raise NotImplementedError("This direction has not been implemented yet.")

# Must be here to prevent a circular import.
import operators
from operators import BaseOperators

# May want to add in support to prevent mixed precision operations (i.e. fp16 and fp32).
# Either that or need to typecast them along the way.
class Tensor():
    def __init__(self, data, operator = None):
        if data.__class__ == LazyTensor:
            self.data = data
        else:
            # Find a better way of doing this without importing the operators.
            # There could be a creation method in the class, which moves it there.
            # Would also prevent importing the lazy operator.
            self.data = LazyTensor(data, LazyOperator(BaseOperators.INSTANTIATE, ()), load = True)
        # When trying to do a topological search, I run into an issue where root nodes don't work
        # because there is no operator. -> Implement istantiation.
        self.operator = operator
        # These are here for when backpropagation gets implemented.
        # self.requires_gradient = requires_gradient
        # self.gradient = None

    ######## UNARY OPERATORS ########

    ######## BINARY OPERATORS ########

    def add(self, x) -> 'Tensor':
        return operators.Addition.apply(self, x)
    
    def sub(self, x) -> 'Tensor':
        return operators.Subtraction.apply(self, x)
    
    # May want to create the matrix in a different manner.
    #### On a forwards pass create it so that it must be transposed before 
    #### being multiplied.
    ######## https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/2
    def matmul(self, x) -> 'Tensor':
        return operators.MatrixMultiplication.apply(self, x)