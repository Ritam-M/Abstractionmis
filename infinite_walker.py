import gym
import numpy as np
import math
import pdb
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym import logger

class InfiniteWalker2d(Walker2dEnv):
    def __init__(self):
        super(InfiniteWalker2d, self).__init__(exclude_current_positions_from_observation=False)
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]
        self.abs_state_dims = 3
        logger.set_level(50)

    def reset(self):
        obs = super(InfiniteWalker2d, self).reset()
        return obs

    def step(self, a):
        state = self.state_vector()
        x = state[0:1]
        dist = np.linalg.norm(x)
        ctrl_cost = self.control_cost(a)
        is_healthy = self.is_healthy

        reward = dist - ctrl_cost + int(is_healthy)
        
        self.do_simulation(a, self.frame_skip)
         
        if not is_healthy:
            ob = self.reset()
        else:
            ob = self._get_obs()
        return ob, reward, False, {}

    def get_initial_state_samples(self, num_states):
        init_states = []
        for i in range(num_states):
            state = self.reset()
            init_states.append(state)
        return np.array(init_states)

    # abstraction related
    def phi(self, state):
        #vec = np.concatenate((state[0:3], state[-3:]))
        vec = state[0:3]
        return vec 

