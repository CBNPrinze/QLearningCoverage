import re
import numpy as np

def take_action(action, current_state, energy_budget, max_energy, violation):
    # Get the current location of the robot
    current_location = tuple(np.argwhere(current_state[2] == 1)[0])
    #print(i," current: ",current_location)
    #print(np.shape(current_state))
    # Define the possible movements based on the action
    movements = {0: (-1, 0),  # Up
                 1: (1, 0),   # Down
                 2: (0, -1),  # Left
                 3: (0, 1)}   # Right

    # Get the movement corresponding to the chosen action
    move = movements.get(action, (0, 0))

    # Calculate the new location after the action
    new_location = tuple(np.add(current_location, move))
    #print("new: ",new_location)

    # Check if the new location is within the environment boundaries
    N = current_state.shape[1]
    new_location = (max(0, min(new_location[0], N - 1)), max(0, min(new_location[1], N - 1)))

    # Check if the new location is an obstacle
    if current_state[0, new_location[0], new_location[1]] == 1:
        # If the new location is an obstacle, stay in the current location
        new_location = current_location

    # Update the current location channel in the state
    next_state = np.copy(current_state)
    next_state[2] = np.zeros((N, N))
    next_state[2][new_location] = 1
    next_state[3][new_location] = 1
    # Calculate energy consumption for the movement (you can adjust this based on your environment)
    energy_consumption = 1  # Example energy consumption for a movement

    # Update the energy budget
    updated_energy_budget = energy_budget - energy_consumption

    # Check if the new location contains a charging station
    if len(current_state) >= 2 and current_state[1].ndim == 2:
        if current_state[1, new_location[0], new_location[1]] == 1:
        # Recharge the energy at the charging station
            updated_energy_budget = max_energy

    # Check if the robot has enough energy to continue
    if updated_energy_budget <= 0:
        # If the energy is depleted, set a negative reward
        reward = -3.0
        violation = 1
    else:
        reward = 0.1  # No reward for other movements
        
    if current_state[3, new_location[0], new_location[1]] == 1:
        reward -=1
    else:
        reward +=2
        
    reward /= 2
    return next_state, reward, updated_energy_budget, violation
