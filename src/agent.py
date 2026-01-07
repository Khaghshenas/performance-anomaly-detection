import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# -----------------------------
# Neural Network for Q-values
# -----------------------------
class QNetwork(nn.Module):
    """
    Simple feedforward neural network for approximating Q-values.
    Input: state vector
    Output: Q-values for each action (2 actions: normal / anomaly)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 actions: normal / anomaly
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -----------------------------
# Deep Q-Learning Agent
# -----------------------------
class DQNAgent:
    """
    Deep Q-Learning agent for anomaly detection.
    Uses epsilon-greedy exploration and experience replay.
    """
    def __init__(
        self,
        state_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995
    ):
        self.state_size = state_size
        self.gamma = gamma  # discount factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay memory
        self.memory = deque(maxlen=50_000)

        # Neural network model and optimizer
        self.model = QNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)  # explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # exploit

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32):
        """
        Sample a minibatch from memory and train the network.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            # Compute target Q-value
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            current_q = self.model(state_tensor)[action]

            # Compute loss and update model
            loss = self.loss_fn(current_q, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

