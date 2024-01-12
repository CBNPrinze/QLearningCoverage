import torch
import random
import numpy as np
# Epsilon-greedy exploration
def select_action(model, state, epsilon, action_size):
    state = torch.tensor(state, dtype=torch.float32)  # Convert state to float tensor
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            q_values = model(state)
            return np.argmax(q_values.numpy())