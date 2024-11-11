import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

grid_size = 5
num_states = grid_size * grid_size

#defining actions

actions = ["up", "down", "right", "left"]

#function to get the next state and reward 

reward_default = -1
high_reward_states = {(2,2): 10, (4,4): 5}

def get_next_state_and_reward(state, action):

    row, col = state

    if action == "up":
        row = max(row - 1, 0)

    if action == "right":
        col = min(col + 1, grid_size - 1)

    if action == "left":
        col = max(col - 1, 0)

    if action == "down":
        row = min(row + 1, grid_size - 1)

    new_state = (row, col)
    reward = high_reward_states.get(new_state, reward_default)

    return new_state, reward




# Discount factor and convergence threshold

gamma = 0.9 
theta = 1e-6

#initialize the equiprobable policy and value function

policy = np.ones((grid_size, grid_size, len(actions))) / len(actions)
value_function = np.zeros((grid_size, grid_size))  # A 5x5 matrix initialized with zeros

def policy_evaluation(policy, value_function):

    while True:
        delta = 0
        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)

                if state in high_reward_states:
                    value_function[row, col] = high_reward_states[state]  # Keep it at the fixed reward
                    continue

                v = 0           #for storing the v in each iteration 

                for action_index, action in enumerate(actions):
                    next_state, reward = get_next_state_and_reward(state, action)
                    next_row, next_col = next_state
                    v += policy[row, col, action_index] * (reward + gamma * value_function[next_row, next_col])

                delta = max(delta, abs(v - value_function[row, col]))
                value_function[row, col] = v

        if delta < theta:
            break

    return value_function



def visualize_value_function(value_function):
    plt.figure(figsize=(6, 6))
    sns.heatmap(value_function, annot=True, fmt=".2f", cmap="YlGnBu", square=True, cbar=True)
    plt.title("Value Function Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()  # So row 0 is at the top
    plt.show()


value_function = policy_evaluation(policy, value_function)

print("value function after convergence: ")
visualize_value_function(value_function)