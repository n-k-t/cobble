from trackers import FormTracker, GlobalTracker

class Buffer:
    def __init__(self, buffer):
        self.buffer = buffer
        self.form_tracker = FormTracker(self.buffer)
        self.data_type = buffer.dtype
        self.size = buffer.size
        self.memory = self.size * 4 # This makes the assumption of fp32.
        GlobalTracker.memory_used += self.memory

    def __del__(self):
        GlobalTracker.memory_used -= self.memory
