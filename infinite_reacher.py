import gym
import numpy as np
import math
import pdb
from gym.envs.mujoco.reacher import ReacherEnv
from gym import logger

class InfiniteReacher(ReacherEnv):
    def __init__(self):
        super(InfiniteReacher, self).__init__()
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]
        self.abs_state_dims = 4
        logger.set_level(50)

    def step(self, action):
        obs, reward, done, info = super(InfiniteReacher, self).step(action)
        if hasattr(self, 'np_random') and self.np_random.random_sample() < 0.03:
            obs = self.reset()
        return obs, reward, False, None

    def get_initial_state_samples(self, num_states):
        init_states = []
        for i in range(num_states):
            state = self.reset() 
            init_states.append(state)
        return np.array(init_states)

    # abstraction related
    def phi(self, state):
        #j1 = np.arctan2(state[2], state[0])
        #j2 = np.arctan2(state[3], state[1])
        ang_v1 = state[6]
        ang_v2 = state[7]
        #vec = np.concatenate(([state[4], state[5]], state[-3:-1]))
        vec = np.concatenate(([ang_v1, ang_v2], state[-3:-1]))
        #abs_s = np.zeros(self.abs_state_dims) 
        #abs_s[idx] = 1#dist
        return np.array(vec)

