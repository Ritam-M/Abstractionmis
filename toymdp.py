import numpy as np
import math
import pdb

class ToyMDP(object):

    def __init__(self, num_states, num_actions, init_state = 0,
            transitions = None, rewards = None):
        self.n_state = num_states
        self.n_action = num_actions
        self.init_state = init_state
        self.state = init_state
        self.transitions = transitions if transitions is not None else np.zeros((num_states, num_actions, num_states + 1))
        self.rewards = rewards if rewards is not None else np.zeros((num_states, num_actions))
        self.n_sa = self.n_state * self.n_action

    def reset(self):
        pass

    def step(self, action):
        states = np.arange(self.n_state + 1)
        n_state = np.random.choice(states,
                                   p=self.transitions[self.state, action])
        rew = self.rewards[self.state, action]
        self.state = n_state
        return n_state, rew, False

    """ getters """
    def get_policy(self, num):
        pass

    def get_init_state(self):
        return self.init_state

    def get_num_states(self):
        return self.n_states

    def get_num_actions(self):
        return self.n_actions

    """ setters """
    def set_transition_prob(self, state, action, n_state, prob):
        self.transitions[state][action][n_state] = prob

    def set_reward(self, state, action, reward):
        self.rewards[state][action] = reward

    def set_transition_probs(self, transition_probs):
        self.transitions = transition_probs

    def set_rewards(self, rewards):
        self.rewards = rewards

    def set_pie(self, pie):
        self.pie = pie

    def set_gamma(self, gamma):
        self.gamma = gamma

# important: MDP assumes absence of terminal state, so agent keeps moving
# until time just runs out
class GraphMDP(ToyMDP):

    def __init__(self, num_states, num_actions, init_state = 0,
            transitions = None, rewards = None, stochastic = False, abstraction = None):
        super(GraphMDP, self).__init__(num_states, num_actions, 
            init_state, transitions, rewards)

        if abstraction:
            self.n_abs_s = abstraction['num_abs_states']
            self.n_abs_sa = self.n_abs_s * self.n_action
            self.phi_map = abstraction['phi']
       
    def step(self, action):
        states = np.arange(self.n_state + 1)
        n_state = np.random.choice(states,
                                   p=self.transitions[self.state, action])
        rew = self.rewards[self.state, action]
        self.state = n_state
        done = False
        return n_state, rew, done, {}

    def reset(self):
        self.state = self.init_state
        return self.state

    def get_initial_state_dist(self):
        # iterate through each box and set the prob equal to number of bins each
        # uniform
        d = {0: 1.} 
        return d

    def get_s_feature_vector(self, encoded_state, abst = False):
        # if abst == True, state is an abstract state, else ground state
        dim = -1
        if abst:
            dim = self.n_abs_s
        else:
            dim = self.n_state
        x = np.zeros(dim)
        x[encoded_state] = 1.
        return x

    def get_feature_vector(self, encoded_state, action, abst = False):
        # if abst == True, state is an abstract state, else ground state
        dim = -1
        if abst:
            dim = self.n_abs_sa
        else:
            dim = self.n_sa
        x = np.zeros(dim)
        x[encoded_state * self.n_action + action] = 1.
        return x

    # abstraction related
    def phi(self, encoded_g_state):
        abs_state = self.phi_map[encoded_g_state]
        #vec = np.zeros(self.n_abs_s)
        #vec[abs_state] = 1.
        return abs_state 

