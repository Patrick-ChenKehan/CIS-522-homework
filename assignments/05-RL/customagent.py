import gymnasium as gym
import numpy as np

# Write a custom agent for the lunar lander player RL problem


class Agent:
    """Agent is a custom agent for the lunar lander player RL problem."""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((20,) * observation_space.shape[0] + (action_space.n,))
        self.last_state = None
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """act returns an action based on the given observation.

        Args:
            observation (gym.spaces.Box): observations from the environment

        Returns:
            gym.spaces.Discrete: action to take
        """
        state = self._discretize(observation)
        action = np.argmax(self.q_table[state])
        self.last_state = state
        self.last_action = action
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Learn from the given observation, reward, and termination status.

        Args:
            observation (gym.spaces.Box): observations from the environment
            reward (float): reward from the environment
            terminated (bool): termination status from the environment
            truncated (bool): truncation status from the environment
        """
        # state = self._discretize(observation)
        # next_state = None

        state = self._discretize(observation)

        self.q_table[self.last_state + (self.last_action,)] += 0.1 * (
            reward
            + 0.99 * np.max(self.q_table[state])
            - self.q_table[self.last_state + (self.last_action,)]
        )

    def _discretize(self, observation: gym.spaces.Box) -> int:
        bins = [
            np.linspace(
                self.observation_space.low[i], self.observation_space.high[i], num=20
            )
            for i in range(self.observation_space.shape[0])
        ]
        state = tuple(
            np.digitize(observation[i], bins[i]) - 1
            for i in range(self.observation_space.shape[0])
        )
        return state
