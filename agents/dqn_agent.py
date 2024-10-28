# Add save and load methods to save the model as .pt files
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self.t_build_model()  # Target network for more stable learning
        self.memory = deque(maxlen=2000)  # Replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),  # Adjust input size to match state size
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
    import torch.nn.functional as F

    def t_build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),  # Increased number of neurons
            nn.BatchNorm1d(64),  # Batch normalization for more stable learning
            nn.LeakyReLU(),  # LeakyReLU to prevent dead neurons
            nn.Linear(64, 64),  # Deeper network with more neurons
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),  # Additional layer with reduced neurons
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, self.action_size)
        )
        
        # Initialize the last layer to have small weights to prevent large initial Q-values
        nn.init.uniform_(model[-1].weight, -0.003, 0.003)
        nn.init.constant_(model[-1].bias, 0)
        
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploitation

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Target for current state
            target = self.model(state).clone()

            if done:
                target[0][action] = reward
            else:
                with torch.no_grad():
                    future_qs = self.target_model(next_state)
                    target[0][action] = reward + self.gamma * torch.max(future_qs)

            output = self.model(state)
            loss = nn.MSELoss()(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=500):
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step([action, action])  # Both actions assumed same
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()

            # Copy weights to the target model every few episodes for stability
            if e % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")

    def save(self, filepath):
        """Save the model as a .pt file"""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Load the model from a .pt file"""
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()  # Set the model to evaluation mode (important for inference)
