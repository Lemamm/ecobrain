import random
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 0.5
        self.learning_rate = 0.001

    def act(self, state):
        return random.randint(0, self.action_size - 1)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        pass  # Simulation légère