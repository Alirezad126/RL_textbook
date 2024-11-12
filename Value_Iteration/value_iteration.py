import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld:
    def __init__(self, grid_size=5, gamma=0.9, theta=1e-4, reward_default=-1, high_reward_states=None):
        self.grid_size = grid_size
        self.actions = ["up", "down", "right", "left"]
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.reward_default = reward_default  # Default reward for movements
        self.high_reward_states = high_reward_states if high_reward_states else {(0,grid_size-1): -3, (grid_size-1,0): 5 }
        
        # Initialize policy and value function as equiprobable policy and zero value
        self.policy = np.ones((grid_size, grid_size, len(self.actions))) / len(self.actions)
        self.value_function = np.zeros((grid_size, grid_size))
        
        # Set initial values for terminal states equal to their associated reward
        for (row, col), reward in self.high_reward_states.items():
            self.value_function[row, col] = reward
    
    def get_next_state_and_reward(self, state, action):
        """Returns the next state and reward for a given action. note that the cooridination is up to down - left to right"""
        row, col = state
        if action == "up":
            row = max(row - 1, 0)
        elif action == "down":
            row = min(row + 1, self.grid_size - 1)
        elif action == "right":
            col = min(col + 1, self.grid_size - 1)
        elif action == "left":
            col = max(col - 1, 0)
        
        new_state = (row, col)
        reward = self.high_reward_states.get(new_state, self.reward_default)
        return new_state, reward

    def value_iteration(self):
        """Evaluates the current policy by updating the value function until convergence."""
        while True:
            delta = 0
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    # Skiping the terminal states
                    if state in self.high_reward_states:
                        continue
                    
                    v = self.value_function[row, col]  # Store old value
                    
                    # Calculate the new value based on the maximum action-value 
                    action_values = np.zeros(len(self.actions))
                    for action_index, action in enumerate(self.actions):
                        next_state, reward = self.get_next_state_and_reward(state, action)
                        next_row, next_col = next_state
                        action_value = reward + self.gamma * self.value_function[next_row, next_col]
                        action_values[action_index] = action_value

                    new_value = max(action_values)
                    
                    # Update value function and delta
                    self.value_function[row, col] = new_value
                    delta = max(delta, abs(v - new_value))
            
            # Check for convergence
            if delta < self.theta:
                optimal_policy = np.zeros((self.grid_size, self.grid_size, len(self.actions)))
                for row in range(self.grid_size):
                    for col in range(self.grid_size):

                        state = (row, col)
                        action_values = np.zeros(len(self.actions))

                        if (row, col) in self.high_reward_states:
                            continue  # Skip terminal states

                        for action_index, action in enumerate(self.actions):
                            next_state, reward = self.get_next_state_and_reward(state, action)
                            next_row, next_col = next_state
                            action_value = reward + self.gamma * self.value_function[next_row, next_col]
                            action_values[action_index] = action_value

                        optimal_action = np.zeros(len(self.actions))
                        optimal_action[np.argmax(action_values)] = 1
                        optimal_policy[row, col] = optimal_action

                break

            
        return optimal_policy
    
    
    def visualize_value_function(self):
        """Displays the value function as a heatmap."""
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.value_function, annot=True, fmt=".2f", cmap="YlOrRd", square=True, cbar=True)
        plt.title("Value Function Heatmap")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()

    def visualize_policy(self, policy):
        """Displays the best action for each state as arrows."""
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.value_function, annot=False, fmt=".2f", cmap="YlOrRd", square=True, cbar=True)
        
        action_symbols = {
            "up": '↑',
            "down": '↓',
            "right": '→',
            "left": '←'
        }
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) in self.high_reward_states:
                    continue  # Skip terminal states

                # Determine the best action for this state
                best_action_index = np.argmax(policy[row, col])
                best_action = self.actions[best_action_index]
                
                # Plot the arrow for the best action
                plt.text(col + 0.5, row + 0.5, action_symbols[best_action], ha='center', va='center', color='black')
        
        plt.title("Policy Visualization with Best Actions")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()


if __name__ == "__main__":
    grid_world = GridWorld()
    policy = grid_world.value_iteration()
    grid_world.visualize_value_function()
    grid_world.visualize_policy(policy)