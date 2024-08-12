import gym
import numpy as np
import math
import pdb
from d4rl.locomotion.ant import AntMazeEnv
from d4rl.locomotion import maze_env
from gym import logger

class InfiniteAntUMaze(AntMazeEnv):
    def __init__(self):
        kwargs = {
            'maze_map': maze_env.U_MAZE_TEST,
            'reward_type':'sparse',
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
            'non_zero_reset':False, 
            'eval':True,
            'maze_size_scaling': 4.0,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0,
            'v2_resets': True,
        }
        super(InfiniteAntUMaze, self).__init__(**kwargs)
        self.fixed_target = (1., 8.8)
        self.set_target(self.fixed_target)
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]
        self.abs_state_dims = 2
        logger.set_level(50)

    def reset(self):
        obs = super(InfiniteAntUMaze, self).reset()
        self.set_target(self.fixed_target)
        return obs

    def step(self, a):
        #self.BASE_ENV.step(self, a)
        if self.reward_type == 'dense':
            reward = -np.linalg.norm(self.target_goal - self.get_xy())
        elif self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(self.get_xy() - self.target_goal) <= 0.5 else 0.0
            
        done = False
        # Terminate episode when we reach a goal
        if np.linalg.norm(self.get_xy() - self.target_goal) <= 0.5:
            done = True
        
        self.BASE_ENV.step(self, a)
        
        if done:
            obs = self.reset()
        else:
            obs = self._get_obs()

        return obs, reward, False, {}

    '''
    def step(self, a):
        obs, reward, done, info = super(InfiniteAntUMaze, self).step(a)
        if done:
            reward = 100.
        #if np.linalg.norm(self.get_xy() - self.target_goal) <= 0.5:
        #    obs = self.reset()
        return obs, reward, False, {}
    '''

    def get_initial_state_samples(self, num_states):
        init_states = []
        for i in range(num_states):
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

            # Set everything other than ant to original position and 0 velocity.
            qpos[15:] = self.init_qpos[15:]
            qvel[14:] = 0.
            state = np.concatenate([qpos[:15], qvel[:14]])
            init_states.append(state)
        return np.array(init_states)

    # abstraction related
    def phi(self, state):
        xy = state[:2]
        in_terminal = int(np.linalg.norm(xy - self.target_goal) <= 0.5)
        #x = int(state[0])
        #y = int(state[1])
        #enc = x + (y * 10)
        abs_s = np.zeros(self.abs_state_dims)
        abs_s[in_terminal] = 1.
        #abs_s[enc] = 1.
 
        return xy#abs_s

