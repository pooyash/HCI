#-------------------------------------------------------------------------------
# Name:        VHCI_Python_Main_Script
# Purpose:     Personal research
# Author:      Pouya
# Created:     09/10/2024
# Copyright:   (c) Pouya 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

class Human:
    def __init__(self, state_size, action_size, gamma=0.95):
        self.state_size = state_size  # Define the size of the state space
        self.action_size = action_size  # Define the size of the action space
        self.gamma = gamma  # Discount factor for future rewards
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.model = self.build_model()  # Build the neural network model

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(1,)))  # Input layer expects a single feature input
        model.add(Dense(24, activation='relu'))  # Hidden layer
        model.add(Dense(24, activation='relu'))  # Hidden layer
        model.add(Dense(self.action_size, activation='linear'))  # Output layer
        model.compile(loss='mse', optimizer='adam')  # Compile the model with mean squared error loss
        return model

    def update_q_table(self, state, action, reward, next_state):
        next_state = np.array(next_state)

        # Ensure next_state is in the correct shape for prediction
        if next_state.ndim == 0:
            next_state = next_state.reshape(1, 1)
        elif next_state.ndim == 1:
            if next_state.size == 1:
                next_state = next_state.reshape(1, 1)
            else:
                raise ValueError("next_state must be a one-dimensional array with one sample.")
        elif next_state.ndim != 2 or next_state.shape[0] != 1 or next_state.shape[1] != 1:
            raise ValueError("next_state must be a 2D array with shape (1, 1).")

        # Calculate the target Q-value
        target = reward + self.gamma * np.max(self.model.predict(next_state, verbose=0))

        # Ensure state and action are integers for indexing
        state = int(state)
        action = int(action)

        # Update Q-table
        self.q_table[state, action] = target

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.reset_environment()  # Reset the environment and get the initial state
            done = False

            while not done:
                action = self.select_action(state)  # Select an action based on the current state
                next_state, reward, done = self.step(action)  # Take action and get the next state, reward, and done status
                self.update_q_table(state, action, reward, next_state)  # Update the Q-table
                state = next_state  # Update the current state

    def reset_environment(self):
        return np.random.randint(0, self.state_size)  # Return a random initial state

    def select_action(self, state):
        return np.random.choice(self.action_size)  # Randomly select an action

    def step(self, action):
        next_state = np.random.randint(0, self.state_size)  # Randomly generate the next state
        reward = 1  # Sample reward
        done = np.random.rand() < 0.1  # Randomly determine if the episode is done
        return next_state, reward, done  # Return the next state, reward, and done status

    def interact(self, num_dialogues=10):
        # Define state and action mappings for better readability
        state_mapping = {
            0: 'State A',
            1: 'State B',
            2: 'State C',
            3: 'State D',
            4: 'State E',
            5: 'State F',
            6: 'State G',
            7: 'State H',
            8: 'State I',
            9: 'State J'
        }
        action_mapping = {
            0: 'Action 1 - Move Forward',
            1: 'Action 2 - Turn Left',
            2: 'Action 3 - Turn Right',
            3: 'Action 4 - Jump',
            4: 'Action 5 - Rest'
        }
        reward_mapping = {
            10: "10 Points - Excellent Move",
            5: "5 Points - Good Move",
            0: "0 Points",
            -5: "-5 Points - Minor Penalty",
            -10: "-10 Points - Poor Decision",
            15: "15 Points - Bonus Reward",
            5: "5 Points - Assist",
            20: "20 Points - Level Up",
            -15: "-15 Points - Major Setback",
            10: "10 Points - Resource Collected"
        }
        for dialogue in range(num_dialogues):
            print(f"*************************************************")
            print(f"Dialogue {dialogue + 1}:")
            print(f"*************************************************")
            state = self.reset_environment()  # Reset the environment for each dialogue
            done = False
            while not done:
                action = self.select_action(state)  # Select an action
                next_state, numerical_reward, done = self.step(action)  # Take the action
                # Map numerical reward to descriptive text
                reward_text = reward_mapping.get(numerical_reward, f"{numerical_reward} Points - No Description")
                # Using descriptive text for states and actions
                print(f"    Current state: {state_mapping[state]}")
                print(f"    Selected action: {action_mapping[action]}")
                print(f"    Next state: {state_mapping[next_state]}")
                print(f"    Reward: {reward_text}")
                state = next_state  # Update the current state
                print(f"=================================================")
            # Separation between dialogues

if __name__ == "__main__":
    state_size = 10  # Define the number of possible states
    action_size = 5   # Define the number of possible actions
    human = Human(state_size, action_size)  # Create an instance of the Human class
    human.train(num_episodes=100)  # Train the model for 100 episodes
    human.interact(num_dialogues=3)  # Execute 10 dialogues with the virtual human

def main():
    pass
if __name__ == '__main__':
    main()