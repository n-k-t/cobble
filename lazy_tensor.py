from buffer import Buffer

class LazyOperator():
    # Get this to deal with multiple inputs.
    def __init__(self, operator, *operand):
        self.operator = operator
        self.operand = set(operand)

# Thinking of moving the shapetracker here and tracking it through the operations even 
# though nothing is executed. The should be doable without strain on the framework.
# Should also add a condition to the tensor class so than on initialization of a weight 
# it does not create a lazytensor (or if it does, the value is already known).
# Track the lazy operator within the lazy tensor --> also allows me to get a tree of the 
# parents.
#### LazyOperator class is different to the operator class --> this should track the base level 
#### operators (what goes on inside the operators).
######## Also track the children?
class LazyTensor():
    def __init__(self, data, lazy_operator, load = False, executed = False):
        if load:
            self.data = Buffer(data)
        else:
            self.data = None
        # This is not working with the base operands at the moment, rethink.
        self.operator = lazy_operator
        self.executed = executed


