from buffer import Buffer

class LazyOperator():
    def __init__(self, base_operator, *operands):
        self.base_operator = base_operator
        self.operands = set(*operands)

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
    def __init__(self, lazy_operator, lazy_data = None, load = False, executed = False):
        self.lazy_operator = lazy_operator
        if load:
            self.lazy_data = Buffer(lazy_data)
        else:
            self.lazy_data = lazy_data
        self.executed = executed

    # Helps create the tree to track the operators as they are applied.
    def new_lazy_tensor(base_operator, *operands):
        return LazyTensor(LazyOperator(base_operator, *operands))


