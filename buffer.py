from trackers import GlobalTracker

class Buffer:
    def __init__(self, buffer) -> 'Buffer':
        self.buffer = buffer
        self.data_type = buffer.dtype
        self.size = buffer.size
        self.memory = self.size * 4 # This makes the assumption of fp32.
        GlobalTracker.memory_used += self.memory

    def __del__(self) -> 'None':
        GlobalTracker.memory_used -= self.memory
