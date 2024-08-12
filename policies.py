import torch
from torch import nn
import numpy as np
import pickle
import pdb

from stable_baselines3 import PPO

class RBFKernel(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma_sq = sigma ** 2

    def __call__(self, x1, x2):
        diff = torch.from_numpy(x1 - x2).float()
        return torch.exp(-torch.sum(torch.square(diff), axis = 1) / (2. * self.sigma_sq))

class PositivityActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _positivity_activation(self, input):
        return torch.square(input)
        #return torch.log(1 + torch.exp(input))

    def forward(self, input):
        return self._positivity_activation(input) 

class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dim = 16, hidden_layers = 1,
                    activation = nn.Tanh(),
                    positivity_constraint = False,
                    batch_norm = False,
                    softmax = False):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.tensor = torch.as_tensor
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.positivity_constraint = positivity_constraint
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.penultimate, self.output = self._create_network()
        self._initialize() 

    def __call__(self, s):
        s = torch.from_numpy(s).float()
        net_out = self.output(s)
        net_out = net_out.detach().numpy()
        return net_out

    def forward(self, s, requires_grad = True):
        s = torch.from_numpy(s).float()
        net_out = self.output(s)
        #if not requires_grad:
        #    net_out = net_out.detach().numpy()
        return net_out

    def get_penultimate(self, s, requires_grad = True):
        s = torch.from_numpy(s).float()
        pen = self.penultimate(s)
        return pen

    def _create_network(self):
        net_arch = []
        curr_dims = self.input_dims
        next_dims = self.hidden_dim
        for l in range(self.hidden_layers):
            net_arch.append(nn.Linear(curr_dims, next_dims))
            if self.batch_norm:
                net_arch.append(nn.BatchNorm1d(next_dims))
            net_arch.append(self.activation)
            curr_dims = next_dims
        
        penultimate = nn.Sequential(*net_arch).float()
        net_arch.append(nn.Linear(curr_dims, self.output_dims))
        if self.positivity_constraint:
            net_arch.append(PositivityActivation())
        if self.softmax:
            net_arch.append(nn.Softmax(dim = -1))
        output = nn.Sequential(*net_arch).float()
        return penultimate, output

    def _initialize(self):
        for m in self.output.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

class NNPolicy:
    def __init__(self, state_dims = None, action_dims = None, f_name = None):
        if f_name is None:
            self.pi = NeuralNetwork(state_dims, action_dims, softmax = True).to('cpu')
            self.state_dims = state_dims
            self.action_dims = action_dims
            self.action_inds = [i for i in range(self.action_dims)]
        else:
            self.load_model(f_name)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr = 1e-4)

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        a_dist = self.pi(s)
        try:
            a = np.random.choice(self.action_inds, p = a_dist)
        except:
            pdb.set_trace()
        prob = a_dist[a]
        return a, prob

    def batch_sample(self, states):
        batch_size = len(states)
        probs = self.pi(states)
        cu = probs.cumsum(axis = 1)
        uni = np.random.rand(batch_size, 1)
        acts = (uni < cu).argmax(axis = 1)
        acts = np.reshape(acts, (batch_size, 1))
        return acts

    def train(self, s, a, G):
        #s = np.reshape(s, (1, len(s)))
        prob = self.pi.forward(s)
        log_prob = torch.log(prob[a])
        obj = -G * log_prob
        self.optimizer.zero_grad()
        obj.backward()

        #nn.utils.clip_grad_value_(self.pi.parameters(), clip_value = 1.0)
        self.optimizer.step()

    def save_model(self, f_name):
        torch.save(self.pi.state_dict(), f_name + '.pth')
        meta_map = {
            'state_dims': self.state_dims,
            'action_dims': self.action_dims,
        }
        np.save(f_name + '.npy', meta_map)

    def load_model(self, f_name):
        meta_map = np.load(f_name + '.npy', allow_pickle = True).item()
        self.state_dims = meta_map['state_dims']
        self.action_dims = meta_map['action_dims']
        self.action_inds = [i for i in range(self.action_dims)]

        self.pi = NeuralNetwork(self.state_dims, self.action_dims, softmax = True).to('cpu')
        self.pi.load_state_dict(torch.load(f_name + '.pth'))

class UniformNNMixPolicy:
    def __init__(self, pi, alphas, discrete_acts = True, act_low = None, act_high = None):
        assert sum(alphas) == 1.

        self.pi = pi
        self.alphas = np.array(alphas)
        self.pi_inds = [0, 1] # [uni, pie NN]
        self.action_dims = None
        self.action_inds = None
        if discrete_acts:
            self.action_dims = pi.action_dims
            self.action_inds = [i for i in range(self.action_dims)]
        self.discrete_acts = discrete_acts
        self.act_low = act_low
        self.act_high = act_high

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        # sample from weighted probabilities to select kth pi
        # sample from kth pi
        k = np.random.choice(self.pi_inds, p = self.alphas)
        if k == 0:
            if self.discrete_acts:
                a = np.random.choice(self.action_inds)
            else:
                a = np.random.uniform(self.act_low, self.act_high) 
        elif k == 1:
            a = self.pi(s)
        return a, None

class GMixPolicy:
    def __init__(self, pis, alphas):
        assert len(pis) == len(alphas)
        assert sum(alphas) == 1.

        self.pis = np.array(pis)
        self.alphas = np.array(alphas)
        self.pi_inds = [i for i in range(len(self.pis))]

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        # sample from weighted probabilities to select kth pi
        # sample from kth pi
        k = np.random.choice(self.pi_inds, p = self.alphas)
        pi_k = self.pis[k]
        a = pi_k(s)
        return a, None

class GaussianPolicy:
    def __init__(self, state_dims = None, action_dims = None, std = None, f_name = None):
        if f_name is None:
            self.pi = NeuralNetwork(state_dims, action_dims).to('cpu')
            self.state_dims = state_dims
            self.action_dims = action_dims
            self.std = std
        else:
            self.load_model(f_name)
        assert (len(self.std) == self.action_dims)
        self.log_std = np.log(self.std)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr = 1e-3)
    
    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        mean = self.get_mean(s)
        if not stochastic:
            return mean, 1.
        # sample action_dims independent samples from N(0, 1)
        # then using respective mean and std for each component, adjust the sample value
        rnd = np.random.normal(size = self.action_dims)
        a = rnd * self.std + mean
        return a, self.pdf(s, a)

    def train(self, s, a, G):
        mean = self.get_mean(s, requires_grad = True)
        a = torch.from_numpy(a).float()
        std = torch.from_numpy(self.std).float()
        log_std = torch.from_numpy(self.log_std).float()
        z = (a - mean) / std
        log_li = -1 * torch.sum(log_std) \
                -0.5 * self.action_dims * np.log(2 * np.pi)\
                -0.5 * torch.sum(torch.square(z))
        neg_loss = -G * log_li
        
        self.optimizer.zero_grad()
        neg_loss.backward()
        self.optimizer.step()

    def log_li(self, mean, a): 
        z = (a - mean) / self.std
        log_li = -1 * np.sum(self.log_std) \
                -0.5 * self.action_dims * np.log(2 * np.pi)\
                -0.5 * np.sum(np.square(z))
        return log_li

    def pdf(self, s, a):
        mean = self.get_mean(s)
        
        # p(x,y) = p(x) * p(y) since independent sampling for each action dimension
        # same mean and std deviation for each dimension
        # summations are along action dimensions
        # TODO revisit for vectorized approach? axis = -1? etc
        prob = np.exp(self.log_li(mean, a))
        return prob

    def get_mean(self, s, requires_grad = False):
        return self.pi.forward(s, requires_grad)

    def save_model(self, f_name):
        torch.save(self.pi.state_dict(), f_name + '.pth')
        meta_map = {
            'state_dims': self.state_dims,
            'action_dims': self.action_dims,
            'std': self.std
        }
        np.save(f_name + '.npy', meta_map)

    def load_model(self, f_name):
        meta_map = np.load(f_name + '.npy', allow_pickle = True).item()
        self.state_dims = meta_map['state_dims']
        self.action_dims = meta_map['action_dims']
        self.std = meta_map['std']

        self.pi = NeuralNetwork(self.state_dims, self.action_dims).to('cpu')
        self.pi.load_state_dict(torch.load(f_name + '.pth'))

class GaussianMixPolicy:
    def __init__(self, pis, alphas):
        assert len(pis) == len(alphas)
        assert sum(alphas) == 1.

        # sum of Gaussians
        self.action_dims = pis[0].action_dims
        self.pis = np.array(pis)
        self.alphas = np.array(alphas)
        self.pi_inds = [i for i in range(len(self.pis))]

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        # sample from weighted probabilities to select kth pi
        # sample from kth pi
        k = np.random.choice(self.pi_inds, p = self.alphas)
        pi_k = self.pis[k]
        mean = pi_k.get_mean(s)

        rnd = np.random.normal(size = self.action_dims)
        a = rnd * pi_k.std + mean
        return a, None

class LeanGridPolicy:
    def __init__(self, x_thr, lean, state_dims = 2, action_dims = 2,
            ds = 0.1, mean = 0, std = 0.05, fwd_decay = -1):
        self.x_thr = x_thr
        self.lean = lean
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.mean = mean
        self.std = std
        self.dir_step = ds
        self.fwd_decay = fwd_decay # example 1. / len(hallway)
        self.min_step_frac = 0.35

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def batch_sample(self, states):
        acts = np.zeros((len(states), self.action_dims))
        acts[:, 1] = self.dir_step * (1. if self.fwd_decay == -1 else np.maximum(self.min_step_frac, np.ceil(states[:, 1]) * self.fwd_decay))
        rnd = np.random.normal(size = (len(states), self.action_dims))
        a_noises = rnd * self.std + self.mean
        xs = states[:, 0]
        if self.lean == 'left':
            inds = np.where(xs > self.x_thr)
            acts[inds, 0] = -self.dir_step
        elif self.lean == 'right':
            inds = np.where(xs < self.x_thr)
            acts[inds, 0] = self.dir_step
        acts += a_noises
        return acts 

    def get_action(self, s, stochastic = True):

        act = np.zeros(self.action_dims)
        rnd = np.random.normal(size = self.action_dims)
        a_noise = rnd * self.std + self.mean
        act[1] = self.dir_step * (1. if self.fwd_decay == -1 else np.maximum(self.min_step_frac, np.ceil(s[1]) * self.fwd_decay))
        
        if self.lean == 'left':
            # if strayed away, bring back
            if s[0] > self.x_thr:
                act[0] = -self.dir_step # move left
        elif self.lean == 'right':
            if s[0] < self.x_thr:
                act[0] = self.dir_step # move right           
        # add noise 
        act += a_noise
        return act, None

class LeanGridMixPolicy:
    def __init__(self, left_x_thr, right_x_thr, alphas,
                ds1 = 0.1, m1 = 0, std1 = 0.05, fwd_decay1 = -1,
                ds2 = 0.1, m2 = 0, std2 = 0.05, fwd_decay2 = -1):
        self.left_x_thr = left_x_thr
        self.right_x_thr = right_x_thr
        self.alphas = alphas
        self.pi_left = LeanGridPolicy(self.left_x_thr, 'left', ds = ds1, mean = m1, std = std1, fwd_decay = fwd_decay1)
        self.pi_right = LeanGridPolicy(self.right_x_thr, 'right', ds = ds2, mean = m2, std = std2, fwd_decay = fwd_decay2)
        self.pis = [self.pi_left, self.pi_right]
        self.pi_inds = [0, 1]

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def get_action(self, s, stochastic = True):
        # sample from weighted probabilities to select kth pi
        # sample from kth pi
        k = np.random.choice(self.pi_inds, p = self.alphas)
        pi_k = self.pis[k]
        act = pi_k(s)
        return act, None

class DiscreteLeanGridPolicy:
    def __init__(self, mdp, x_thr, lean, state_dims = 2, n_action = 4):
        self.mdp = mdp
        self.x_thr = x_thr
        self.lean = lean
        self.state_dims = state_dims
        self.n_action = n_action
        self.actions = [a for a in range(n_action)]
        self.action_probs = [1. / n_action for _ in range(n_action)]

    def __call__(self, s):
        s = self.mdp.state_decoding(s)
        a, _ = self._get_action(s)
        return a

    def get_prob(self, s, a):
        s = self.mdp.state_decoding(s)
        if self.lean == 'left':
            if s[0] >= self.x_thr:
                if a == 2:
                    return 1.
                else:
                    return 0.
        elif self.lean == 'right':
            if s[0] <= self.x_thr:
                if a == 0:
                    return 1.
                else:
                    return 0.
        return 0.25

    def _get_action(self, s, stochastic = True):
        act = np.random.choice(self.actions, p = self.action_probs)
        ret = (act, 0.25)
        if self.lean == 'left':
            # if strayed away, bring back
            if s[0] >= self.x_thr:
                ret = (2, 1.) # move left
        elif self.lean == 'right':
            if s[0] <= self.x_thr:
                ret = (0, 1.) # move right
        return ret

class DiscreteGridworldElbowSlow:
    def __init__(self, mdp, elbow_type, fwd_back_probs = [0.4, 0.4, 0.1, 0.1]):
        self.mdp = mdp
        self.elbow = elbow_type
        self.fwd_back_probs = fwd_back_probs

    def __call__(self, s):
        s = self.mdp.state_decoding(s)
        a = self._get_action(s)
        return a

    def _get_action(self, s):
        x, y = s[0], s[1]
        if self.elbow == 'left':
            if x >= self.mdp.max_x_coord / 2:
                return 2 # go left
        elif self.elbow == 'right':
            if x <= self.mdp.max_x_coord / 2:
                return 0 # go right
        return np.random.choice([1, 3, 2, 0], p = self.fwd_back_probs)

    def get_prob(self, s, a):
        s = self.mdp.state_decoding(s)
        x, y = s[0], s[1]
        if self.elbow == 'left':
            if x >= self.mdp.max_x_coord / 2:
                return 1. if a == 2 else 0 # go left, min out x
        elif self.elbow == 'right':
            if x <= self.mdp.max_x_coord / 2:
                return 1. if a == 0 else 0 # go right, max out y
        if a == 1:
            return self.fwd_back_probs[0]
        elif a == 3:
            return self.fwd_back_probs[1]
        elif a == 2:
            return self.fwd_back_probs[2]
        elif a == 0:
            return self.fwd_back_probs[3]

class DiscreteGridworldElbow:
    def __init__(self, mdp, elbow_type):
        self.mdp = mdp
        self.elbow = elbow_type

    def __call__(self, s):
        s = self.mdp.state_decoding(s)
        a = self._get_action(s)
        return a

    def _get_action(self, s):
        x, y = s[0], s[1]
        if self.elbow == 'left':
            if x >= self.mdp.max_x_coord / 2:
                a = 2 # go left
            else:
                a = 1 # go up
        elif self.elbow == 'right':
            if x <= self.mdp.max_x_coord / 2:
                a = 0 # go right
            else:
                a = 1 # go up
        return a

    def get_prob(self, s, a):
        s = self.mdp.state_decoding(s)
        x, y = s[0], s[1]
        if self.elbow == 'left':
            if x >= self.mdp.max_x_coord / 2:
                return 1. if a == 2 else 0 # go left, min out x
            else:
                return 1. if a == 1 else 0 # go up once x mined out
        elif self.elbow == 'right':
            if x <= self.mdp.max_x_coord / 2:
                return 1. if a == 0 else 0 # go right, max out y
            else:
                return 1. if a == 1 else 0 # go up once y maxed out

class DiscreteGridworldRandomElbow:
    def __init__(self, mdp, elbow_type, alphas):
        self.elbow_pi = DiscreteGridworldElbow(mdp, elbow_type)
        self.alphas = alphas
        self.pi_inds = [0, 1]
        self.num_actions = mdp.n_action

    def __call__(self, s):
        k = np.random.choice(self.pi_inds, p = self.alphas)
        if k == 0:
            # randomly choose:
            return np.random.choice(range(self.num_actions)) 
        elif k == 1:
            return self.elbow_pi(s)

    def get_prob(self, s, a):
        alpha = self.alphas[0]
        random_prob = alpha * (1. / self.num_actions)
        elbow_prob = (1. - alpha) * self.elbow_pi.get_prob(s, a)
        return random_prob + elbow_prob

class AbstractDiscreteGridworld:
    def __init__(self, mdp, pi, g_densities):
        self.mdp = mdp
        self.num_actions = mdp.n_action
        self.pi = pi
        self.g_densities = g_densities
        self.abs_pi = np.zeros((mdp.n_abs_s, mdp.n_action))
        self._compute_abs_pi()

    def _compute_abs_pi(self):
        norms = np.zeros(self.mdp.n_abs_s)
        for s in range(self.mdp.n_state):
            for a in range(self.num_actions):
                val = self.pi.get_prob(s, a) * self.g_densities[s]
                self.abs_pi[self.mdp.phi(s), a] += val
            norms[self.mdp.phi(s)] += self.g_densities[s]
        norms = norms + 1e-10
        self.abs_pi = self.abs_pi / norms[:, None]

    def __call__(self, s):
        s = self.mdp.phi(s)
        try:
            act = np.random.choice(range(self.num_actions), p = self.abs_pi[s] / sum(self.abs_pi[s]))
        except:
            pdb.set_trace()
        return act

    def get_prob(self, s, a):
        return self.abs_pi[s, a]

class DeepOPEPolicy:
    def __init__(self, pkl_file, action_dims, std = 1.):
        with open(pkl_file, 'rb') as f:
            weights = pickle.load(f)
        self.fc0_w = weights['fc0/weight']
        self.fc0_b = weights['fc0/bias']
        self.fc1_w = weights['fc1/weight']
        self.fc1_b = weights['fc1/bias']
        self.fclast_w = weights['last_fc/weight']
        self.fclast_b = weights['last_fc/bias']
        self.fclast_w_logstd = weights['last_fc_log_std/weight']
        self.fclast_b_logstd = weights['last_fc_log_std/bias']
        relu = lambda x: np.maximum(x, 0)
        self.nonlinearity = np.tanh if weights['nonlinearity'] == 'tanh' else relu

        identity = lambda x: x
        self.output_transformation = np.tanh if weights[
            'output_distribution'] == 'tanh_gaussian' else identity
        self.action_dims = action_dims # env.action_space.shape[0]
        self.std = std

    def __call__(self, state):
        action, _ = self.act(state)
        return action

    def act(self, state):
        x = np.dot(self.fc0_w, state) + self.fc0_b
        x = self.nonlinearity(x)
        x = np.dot(self.fc1_w, x) + self.fc1_b
        x = self.nonlinearity(x)
        mean = np.dot(self.fclast_w, x) + self.fclast_b
        logstd = np.dot(self.fclast_w_logstd, x) + self.fclast_b_logstd
        noise = self.std * np.random.randn(self.action_dims).astype(np.float32)
        action = self.output_transformation(mean + np.exp(logstd) * noise)
        return action, mean
    
    def batch_sample(self, states):
        num_samples = len(states)
        x = np.dot(states, self.fc0_w.T) + self.fc0_b
        x = self.nonlinearity(x)
        x = np.dot(x, self.fc1_w.T) + self.fc1_b
        x = self.nonlinearity(x)
        mean = np.dot(x, self.fclast_w.T) + self.fclast_b
        logstd = np.dot(x, self.fclast_w_logstd.T) + self.fclast_b_logstd
        noise = self.std * np.random.randn(num_samples, self.action_dims).astype(np.float32)
        action = self.output_transformation(mean + np.exp(logstd) * noise)
        return action 

class AbsSimPolicy:
    def __init__(self, pi_path, env, deterministic = False):
        self.model = PPO.load(pi_path, env, custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 1.0})
        self.deterministic = deterministic

    def __call__(self, state):
        action, _ = self.act(state)
        return action

    def act(self, state):
        action = self.model.predict(state, deterministic = self.deterministic)[0]
        return action, None
    
    def batch_sample(self, states):
        actions = self.model.predict(states, deterministic = self.deterministic)[0]
        return actions

class ToyMDPPolicy:
    def __init__(self, num_actions, probs):
        self.probs = probs
        self.num_actions = num_actions

    def __call__(self, s, stochastic = True):
        a, _ = self.get_action(s, stochastic)
        return a

    def batch_sample(self, states):

        pdb.set_trace()

        return acts 

    def get_action(self, s, stochastic = True):
        act = np.random.choice(self.num_actions, p = self.probs[s])
        return act, None

    def get_prob(self, s, a):
        return self.probs[s, a]

