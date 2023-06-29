from tensor import Operator
from lazy_tensor import LazyTensor

######## BASE OPERATORS ########

# This is a dict for now allowing us to store functions to call in the 
# future, however should still consider whether or not this is the right 
# approach.

# Dropped the dict and made a class to allow us to create an enum.
# Using a base class instead of an enum is faster on lookup.
# None is smaller than lambda: None

class BaseOperators:
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    POWER = "POWER"
    MATRIX_MULTIPLY = "MATRIX_MULTIPLY"

class Addition(Operator):
    def forward(self, *operands) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(BaseOperators.ADD, operands)
    
class Subtraction(Operator):
    def forward(self, *operands) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(BaseOperators.SUBTRACT, operands)

class MatrixMultiplication(Operator):
    def forward(self, *operands) -> 'LazyTensor':
        return LazyTensor.new_lazy_tensor(BaseOperators.MATRIX_MULTIPLY, operands)
