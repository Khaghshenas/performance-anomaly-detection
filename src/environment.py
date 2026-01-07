import numpy as np
import gym
from gym import spaces

class ServerAnomalyEnv(gym.Env):
    """
    Reinforcement Learning environment for detecting server performance anomalies.

    Each environment instance represents a single server's CPU utilization time series.

    Observations:
        - Sliding window of CPU utilization values of length `window_size`

    Actions:
        - 0: Predict normal
        - 1: Predict anomaly

    Rewards:
        - Designed based on deviation from normal CPU behavior
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, cpu_series: np.ndarray, window_size: int = 12, anomaly_threshold: float = 20):
        """
        Args:
            cpu_series (np.ndarray): Time series of CPU utilization values for one server
            window_size (int): Number of past time steps included in each observation
            anomaly_threshold (float): Threshold to define anomalies in CPU usage
        """
        super().__init__()

        self.cpu_series = cpu_series
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        # Observation space: sliding window of CPU usage values
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(window_size,), dtype=np.float32
        )

        # Action space: 0 = normal, 1 = anomaly
        self.action_space = spaces.Discrete(2)

        # Internal pointer to current step in the time series
        self.current_step = window_size


    def _get_state(self) -> np.ndarray:
        """
        Get the current observation for the agent.

        Returns:
          np.ndarray: Sliding window of CPU utilization values 
                    ending at the current step, length = `window_size`.
        """
        start = self.current_step - self.window_size
        end = self.current_step
        return self.cpu_series[start:end]


    def _is_anomaly(self, current_value: float, prev_value: float) -> bool:
        """
        Determine if the change between two consecutive CPU values
        constitutes an anomaly.

        An anomaly is defined as a sudden spike or drop exceeding
        the configured threshold.

        Args:
          current_value (float): Current CPU utilization
          prev_value (float): Previous CPU utilization

        Returns:
          bool: True if the absolute difference exceeds `anomaly_threshold`
        """
        return abs(current_value - prev_value) >= self.anomaly_threshold


    def step(self, action: int):
        """
        Take an action in the environment and advance one time step.

        Args:
          action (int): Action chosen by the agent.
                      0 = predict normal, 1 = predict anomaly

        Returns:
          state (np.ndarray): Next observation (sliding window of CPU usage)
          reward (float): Reward received for the action
          done (bool): True if the end of the CPU series is reached
          info (dict): Additional info (empty for now)
        """
        prev_val = self.cpu_series[self.current_step - 1]
        curr_val = self.cpu_series[self.current_step]

        # Determine if a true anomaly exists
        true_anomaly = self._is_anomaly(curr_val, prev_val)

        # Reward logic
        if action == 1 and true_anomaly:       # Correctly predicted anomaly
            reward = 1.0
        elif action == 1 and not true_anomaly: # False positive
            reward = -0.5
        elif action == 0 and not true_anomaly: # Correctly predicted normal
            reward = 0.5
        else:                                  # Missed anomaly
            reward = -1.0

        # Advance time step
        self.current_step += 1
        done = self.current_step >= len(self.cpu_series)

        # Get next observation
        state = self._get_state() if not done else np.zeros(self.window_size, dtype=np.float32)

        return state.astype(np.float32), reward, done, {}


    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Returns:
          np.ndarray: The initial observation (first sliding window of CPU usage)
        """
        self.current_step = self.window_size
        return self._get_state().astype(np.float32)


    def render(self, mode: str = "human"):
        """
        Optional method for visualizing the environment.

        Currently not implemented. Can be extended to plot CPU usage,
        highlight anomalies, or show agent decisions.
    
        Args:
          mode (str): Rendering mode, default is 'human'
        """
        pass

