"""Implements different OPE methods."""
from __future__ import print_function
from __future__ import division

import pdb
import numpy as np
from utils import get_CI

class Estimator(object):

    def __init__(self):
        pass

    def estimate(self, paths):
        pass

class OnPolicy(Estimator):

    def __init__(self, pi, gamma):
        self.pi = pi
        self.gamma = gamma

    def estimate(self, paths):
        m = len(paths) # number of trajectories
        total = 0
        total_normalization = 0
        for path in paths:
            obs = path['obs']
            acts = path['acts']
            rews = path['rews']
            accum_gamma = 1.
            for t in range(len(obs)):
                o = obs[t]
                a = acts[t]
                r = rews[t]
                total += accum_gamma * r
                total_normalization += accum_gamma
                accum_gamma *= self.gamma
        return total / total_normalization 

class Dice(Estimator):

    def __init__(self, ratios):
        self.ratios = ratios

    def estimate(self, transition_tuples, rewards, gammas, temp = None):
        ratios = self.ratios(transition_tuples)
        ratios = ratios.reshape((len(ratios),))
        print (get_CI(ratios))
        out = np.sum(ratios * rewards) / np.sum(ratios)
        return out

class DiscreteDice(Estimator):

    def __init__(self, ratios):
        self.ratios = ratios

    def estimate(self, data, rewards, gammas):
        ratios = self.ratios[np.argmax(data, axis = 1)]
        out = np.mean(ratios * rewards)
        return out

