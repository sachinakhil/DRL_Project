# Cell 1: Environment Definition (traffic_env.py)

import numpy as np
import gym
from gym import spaces

class TrafficEnv:
    def __init__(self):
        self.signal_NS = 1  # NS signal starts as green
        self.signal_EW = 0  # EW signal starts as red
        self.cars_NS = np.random.randint(0, 10)  # Cars waiting in NS direction
        self.cars_EW = np.random.randint(0, 10)  # Cars waiting in EW direction

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), 
                                            high=np.array([1, 1, np.inf, np.inf]), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2])  # Two traffic lights, each can be red (0) or green (1)
        self.time_since_last_switch = 0
        self.state = self.get_state()

    def get_state(self):
        # Get the current state
        state = np.array([self.signal_NS, self.signal_EW, self.cars_NS, self.cars_EW])
        return state

    def reset(self):
        self.cars_NS = np.random.randint(0, 10)
        self.cars_EW = np.random.randint(0, 10)
        self.signal_NS = 1
        self.signal_EW = 0
        self.time_since_last_switch = 0
        return self.get_state()

    def step(self, action):
        reward = 0
        self.time_since_last_switch += 1
        ns_action, ew_action = action

        if ns_action == 1:
            self.signal_NS = 1
            self.signal_EW = 0
        elif ew_action == 1:
            self.signal_NS = 0
            self.signal_EW = 1

        if self.signal_NS:
            reward += self.cars_NS
            self.cars_NS = max(0, self.cars_NS - np.random.randint(1, 3))

        if self.signal_EW:
            reward += self.cars_EW
            self.cars_EW = max(0, self.cars_EW - np.random.randint(1, 3))

        self.cars_NS += np.random.randint(0, 3)
        self.cars_EW += np.random.randint(0, 3)

        # Ensure state consistency (keep it size 4)
        self.state = np.array([self.cars_NS, self.cars_EW, self.signal_NS, self.signal_EW])
        done = self.time_since_last_switch > 100
        return self.state, reward, done, {}

    def update_traffic_lights(self, action, direction):
        if direction == 'NS':
            self.signal_NS = action
            self.signal_EW = 1 - action
        elif direction == 'EW':
            self.signal_EW = action
            self.signal_NS = 1 - action
