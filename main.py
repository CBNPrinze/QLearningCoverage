from calendar import c
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from image import initialize_environment
from epsilonGreedy import select_action
from action import take_action
from Buffer import ReplayBuffer
from qnetwork import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 16  # Environment size
action_size = 4  # Assuming 4 actions: Up, Down, Left, Right
max_energy = 80  # Maximum energy budget
# Training parameters
input_channels = 4
output_size = action_size # Assuming 4 actions: Up, Down, Left, Right
batch_size = 64
gamma = 0.9
learning_rate = 0.001
epsilon = 0.9
epsilon_decay = 2100
tau = 20  # Target network update frequency
max_episodes = 10000
max_steps_per_episode = 2560

# Initialize networks and optimizer
policy_net = QNetwork(input_channels, output_size, N).to(device)
target_net = QNetwork(input_channels, output_size, N).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Experience replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
for episode in range(max_episodes):
    state = initialize_environment(N)  # Your function to initialize the environment
    total_reward = 0
    violation = 0
    if (episode >= 800):
        #epsiilon giữ nguyên bằng 0.9 từ 0 đên 800 bắt đầu giảm theo hàm e mũ và bằng 0.09 khi episode =7000
        epsilon = 0.09 + (0.9 - 0.09) * np.exp(-1. * (episode - 800) / epsilon_decay)
    budget = max_energy  # Initial energy budget
    for step in range(max_steps_per_episode):
        action = select_action(policy_net, state, epsilon, action_size)
        next_state, reward, budget, violation = take_action(action, state, budget, max_energy, violation)  # Your function to execute action and get next state, reward
        replay_buffer.push((state, action, reward, next_state))
        if len(replay_buffer.buffer) > batch_size:
            
            batch = replay_buffer.sample(batch_size)
            state, actions, rewards, next_state = zip(*batch)

            state = torch.FloatTensor(np.stack(state))
            next_state = torch.tensor(np.stack(next_state), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            q_values = policy_net(state).gather(1, actions)
            next_q_values = target_net(next_state).max(1)[0].detach()

            target_values = rewards + gamma * next_q_values.unsqueeze(1)

            loss = nn.MSELoss()(q_values, target_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reward += reward
            state = next_state[batch_size-1].detach().cpu().numpy()
            #print(np.shape(state))
        else: 
            total_reward += reward
            state = np.copy(next_state)
            #print(np.shape(state))
            
        if step % tau == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        covered_locations = np.logical_and(state[0] == 1, state[3] == 1)
        if (covered_locations.sum() == N * N and violation == 0):
            total_reward += 200
            break

    print(f"Episode {episode}, Total Reward: {total_reward}")
    with open('output.txt', 'a') as file:
        file.write(f"Episode {episode}, Total Reward: {total_reward}\n")
    
    