from tracker import Tracker

class Buffer:
    def __init__(self, buffer, operator = None):
        self.buffer = buffer
        self.operator = operator
        self.data_type = buffer.dtype
        self.size = buffer.size
        self.memory = self.size * 4
        Tracker.memory_used += self.memory

    def __del__(self):
        Tracker.memory_used -= self.memory
