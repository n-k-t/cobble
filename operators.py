from tensor import Operator
from lazy_tensor import LazyTensor

######## BASE OPERATORS ########

# This is a dict for now allowing us to store functions to call in the 
# future, however should still consider whether or not this is the right 
# approach.

# Dropped the dict and made a class to allow us to create an enum.
# Using a base class instead of an enum is faster on lookup.
# None is smaller than lambda: None

class BaseOperators():
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    POWER = "POWER"
    INSTANTIATE = "INSTANTIATE" # For the loading/creation operation at the head of a network/for the weights?
    # This should probably be tracked/better allow integration into the lazy tensor.
    MATRIX_MULTIPLY = "MATRIX_MULTIPLY"


# Add as many data attributes of the tensor as the user can handle.
class Addition(Operator):
    def forward(self, *args) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(operator = BaseOperators.ADD, data = args)
    
class Subtraction(Operator):
    def forward(self, *args) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(operator = BaseOperators.SUBTRACT, data = args)

class MatrixMultiplication(Operator):
    def forward(self, *args) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(operator = BaseOperators.MATRIX_MULTIPLY, data = args)
