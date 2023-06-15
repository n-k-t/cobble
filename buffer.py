from tracker import Tracker

class Buffer:
    def __init__(self, buffer):
        self.buffer = buffer
        self.data_type = buffer.dtype
        self.size = buffer.size
        self.memory = self.size * 4
        Tracker.memory_used += self.memory

    def __del__(self):
        Tracker.memory_used -= self.memory