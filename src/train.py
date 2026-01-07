import os
import pandas as pd
import numpy as np
from src.environment import ServerAnomalyEnv
from src.agent import DQNAgent

# Directory containing PlanetLab CPU usage traces
DATA_DIR = "data/planetlab/20110303"

# Number of time steps per observation window
WINDOW_SIZE = 12

# Training episodes
EPISODES = 80


def load_server_data():
    """
    Load CPU utilization time series from PlanetLab server files.

    Each file represents one server and contains a single CPU utilization
    value per line. Empty lines are ignored.

    Returns:
        List[np.ndarray]: A list of CPU usage time series, one per server.
    """
    cpu_series = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, filename)

        with open(file_path, "r") as f:
            values = [
                float(line.strip())
                for line in f
                if line.strip()
            ]

        cpu_series.append(np.asarray(values))

    return cpu_series

def train_agent(cpu_series, server_id):
    env = ServerAnomalyEnv(cpu_series, window_size=WINDOW_SIZE)
    agent = DQNAgent(state_size=WINDOW_SIZE)

    episode_rewards = []

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Server {server_id} | Episode {ep + 1} | Reward: {total_reward:.2f}")

    return agent, episode_rewards


if __name__ == "__main__":
    # Load CPU utilization traces for all servers
    cpu_datasets = load_server_data()

    trained_agents = []

    for server_id, cpu_series in enumerate(cpu_datasets):
        print(f"\n=== Training agent for server {server_id} ===")
        agent, reward_history = train_agent(cpu_series, server_id)
        trained_agents.append(agent)

    print("\nTraining completed for all servers.")
