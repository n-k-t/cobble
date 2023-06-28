class GlobalTracker:
    memory_used = 0

class FormTracker:
    def __init__(self, data):
        self.shape = data.shape
        self.stride = self.get_stride()
    
    # Get the strides associated with the shape of the tensor. Because arrays/tensors 
    # are stored linearly in memory, the stride allows us to determine the number of 
    # indices (memory addresses) are required to hop in order to index to a specific 
    # element within the newly shaped tensor.
    def get_stride(self):
        stride = [1]

        for i in self.shape[::-1][:-1]:
            stride = [i * stride[0]] + stride

        return stride