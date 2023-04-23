import gymnasium as gym
import numpy as np

# Write a custom agent for the lunar lander player RL problem


class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((20,) * observation_space.shape[0] + (action_space.n,))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        state = self._discretize(observation)
        return np.argmax(self.q_table[state])

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        state = self._discretize(observation)
        next_state = None
        action = self.act(observation)
        if not terminated and not truncated:
            next_state = self._discretize(observation)
            self.q_table[state + (action,)] += 0.1 * (
                reward
                + 0.9 * np.max(self.q_table[next_state])
                - self.q_table[state + (action,)]
            )
        else:
            print(terminated, truncated, reward)
            self.q_table[state + (action,)] += 0.1 * (
                reward - self.q_table[state + (action,)]
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
