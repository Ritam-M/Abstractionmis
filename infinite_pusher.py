import gym
import numpy as np
import math
import pdb
from gym.envs.mujoco.pusher import PusherEnv
from gym import logger

class InfinitePusher(PusherEnv):
    def __init__(self):
        super(InfinitePusher, self).__init__()
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]
        self.abs_state_dims = 6#9
        logger.set_level(50)

    def step(self, action):
        obs, reward, done, info = super(InfinitePusher, self).step(action)
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
        #vec = state[6:-1]
        #vec = np.concatenate([state[4:6], state[-3:-1]])
        tip_vec = state[14:17]
        obj_vec = state[17:20]
        goal_vec = state[20:]
        #vec = state[-14:]
        vec = np.concatenate([obj_vec - tip_vec, obj_vec - goal_vec])
        #dist = np.linalg.norm(vec)
        #abs_s = np.zeros(self.abs_state_dims) 
        #abs_s[0] = dist
        return np.array(vec)

