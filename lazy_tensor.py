from buffer import Buffer
from trackers import FormTracker

class LazyOperator:
    def __init__(self, base_operator, *operands):
        self.base_operator = base_operator
        self.operands = set(*operands)

#### LazyOperator class is different to the operator class --> this should track the base level 
#### operators (what goes on inside the operators).
######## Also track the children?
class LazyTensor:
    def __init__(self, lazy_operator, lazy_data = None, shape_tracker = None, executed = False, load = False):
        self.lazy_operator = lazy_operator
        if load:
            self.lazy_data = Buffer(lazy_data)
            self.shape_tracker = FormTracker(self.lazy_data.buffer)
        else:
            self.lazy_data = lazy_data
            # I need to figure out how I want to integrate this so that the shape is inherited 
            # or modified smoothly.
            self.shape_tracker = shape_tracker
        self.executed = executed

    # Helps create the tree to track the operators as they are applied.
    def new_lazy_tensor(base_operator, *operands):
        return LazyTensor(LazyOperator(base_operator, *operands))


