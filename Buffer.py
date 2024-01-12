import random

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)