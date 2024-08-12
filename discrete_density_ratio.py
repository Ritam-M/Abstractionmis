import estimators
import pdb
import torch
from torch import nn
import numpy as np
from policies import NeuralNetwork
import copy

import pdb

class TabularStateCOPTD(object):
    def __init__(self, name, gamma, mdp):
        self.name = name
        self.gamma = gamma
        self.mdp = mdp

    def compute(self, data, alpha = 1e-3):
        ab = False
        if 'abs' in self.name:
            dims = self.mdp.n_abs_s
            curr_states = data['abs_state_b']
            next_states = data['abs_next_state_b']
            ab = True
        elif self.name == 'ground':
            dims = self.mdp.n_state
            curr_states = data['state_b']
            next_states = data['next_state_b']

        policy_ratios = data['policy_ratios']

        # TODO randomy intiialize to prob dist
        self._c = np.zeros([dims])#np.random.uniform(0, 1, size = dims) 
        #self._c = np.random.uniform(0, 1, size = dims) 
        self._c[0] = 1.
        self._prev_c = np.zeros([dims])
        
        for itr in range(10000):
            for idx, _ in enumerate(curr_states):
                curr_s_idx = np.argmax(curr_states[idx])
                next_s_idx = np.argmax(next_states[idx])
                bootstrap = self._c[curr_s_idx]
                #bootstrap = self._c
                #if curr_s_idx == 0:
                #    bootstrap = 1.
                #err = self.gamma * policy_ratios[idx] * bootstrap * curr_states[idx] + (1. - self.gamma) - next_states[idx] * self._c
                #self._c = self._c + alpha * err
                err = (self.gamma * policy_ratios[idx] * bootstrap) + (1. - self.gamma) - self._c[next_s_idx]
                self._c[next_s_idx] = self._c[next_s_idx] + alpha * err

            #self._c = self._c / np.sum(self._c)
            if (itr + 1) % 100 == 0:
                print (itr + 1)
                print ('actual {}'.format(self._c))
                print ('weighted {}'.format(self._c / np.sum(self._c)))
                diff = np.abs(self._c - self._prev_c)
                count = (diff <= 1e-3).sum()
                print ('norm {}'.format(np.linalg.norm(self._c)))
                #if count == dims:
                #    break
            self._prev_c = copy.deepcopy(self._c)
            if (itr + 1) % 100 == 0:
                alpha = alpha / 2.
        self._c = np.clip(self._c, a_min = 0, a_max = None) 
        #print (self._c)
        weighted = self._c / np.sum(self._c)
        print (weighted)
        print (np.std(weighted))

    def get_W(self):
        return self._c



    
class TabularDualDice(object):
    # adapted from: https://github.com/google-research/dice_rl/blob/master/estimators/tabular_dual_dice.py
    def __init__(self, name, gamma, mdp, pie):
        self.gamma = gamma
        self.name = name
        self.mdp = mdp
        self.pie = pie

    def compute(self, data, eps = 1e-10):

        ab = False
        if 'abs' in self.name:
            q_curr_inputs = data['abs_state_b_act_b']
            dims = self.mdp.n_abs_sa
            ab = True
        elif self.name == 'ground': 
            q_curr_inputs = data['state_b_act_b']
            dims = self.mdp.n_sa

        next_abs_states = data['abs_next_state_b']
        next_g_states = data['next_state_b']

        init_states = data['init_states']

        self._nu = np.zeros([dims])
        self._zeta = np.zeros([dims])
        td_residuals = np.zeros([dims, dims])
        total_weights = np.zeros([dims])
        initial_weights = np.zeros([dims])

        for idx, _ in enumerate(q_curr_inputs):
            curr_idx = np.argmax(q_curr_inputs[idx])
            td_residuals[curr_idx, curr_idx] += 1.
            total_weights[curr_idx] += 1.

            next_abs_state = np.argmax(next_abs_states[idx])
            next_g_state = np.argmax(next_g_states[idx])
            next_state = next_g_state
            
            if ab:
                next_state = next_abs_state 
            for a in range(self.mdp.n_action):
                n_idx = next_state * self.mdp.n_action + a
                td_residuals[n_idx, curr_idx] += -self.pie.get_prob(next_g_state, a) * self.gamma

            for init_s in init_states:
                for a in range(self.mdp.n_action):
                    idx = init_s * self.mdp.n_action + a
                    initial_weights[idx] += 1. * self.pie.get_prob(init_s, a)    

        td_residuals /= np.sqrt(eps + total_weights)[None, :]
        td_errors = np.dot(td_residuals, td_residuals.T)
        self._nu = np.linalg.solve(td_errors + eps * np.eye(dims), (1 - self.gamma) * initial_weights)
        self._zeta = np.dot(self._nu, td_residuals) / np.sqrt(eps + total_weights)
        self.W = self._zeta

    def get_W(self):
        return self.W

class DiscreteDensityRatioEstimation:
    def __init__(self, name, gamma):
        self.gamma = gamma
        self.name = name

    def compute(self, data, eps = 1e-10, print_log = False):

        if 'abs' in self.name:
            w_f = data['abs_state_b_act_b']
            q_curr_inputs = data['abs_state_b_act_b']
            q_next_state_inputs = data['exp_abs_next_state_b_act_e']
            q_init_state_inputs = data['abs_init_state_act_e']
            if 'pract' in self.name:
                q_next_state_inputs = data['exp_abs_next_state_b_act_e_pract']
        elif self.name == 'ground': 
            w_f = data['state_b_act_b']
            q_curr_inputs = data['state_b_act_b']
            q_next_state_inputs = data['exp_next_state_b_act_e']
            q_init_state_inputs = data['init_state_act_e'] 

        dims = len(q_init_state_inputs)
        
        num_samples = data['num_samples']
        td_err_accum = np.zeros((dims, dims))
        init_accum = np.zeros(dims)
        for idx, _ in enumerate(w_f):
            td_err_accum += (np.outer(q_curr_inputs[idx] - self.gamma * q_next_state_inputs[idx], w_f[idx]))

        assert len(td_err_accum) == dims and len(td_err_accum[0]) == dims
        td_err_accum = (td_err_accum / num_samples) + eps * np.eye(dims)
        w = np.dot(np.linalg.inv(td_err_accum), (1. - self.gamma) * q_init_state_inputs)
        self.w = w
        return w 

    def get_W(self):
        return self.w

''' Misc code below '''

class DiscreteFQE:
    def __init__(self, method, mdp, gamma, pie, abs_pie):
        self.method = method
        self.mdp = mdp
        self.gamma = gamma
        self.pie = pie
        self.abs_pie = abs_pie

    def compute(self, data, alpha = 0.5):
        abs_q = True if 'abs' in self.method else False

        q_dims = self.mdp.n_state * self.mdp.n_action
        trans_tups = data['ground_trans']
        if abs_q:
            trans_tups = data['abs_trans']
            q_dims = self.mdp.n_abs_s * self.mdp.n_action
        q_func = np.zeros((q_dims))
        prev_q_func = np.zeros_like(q_func)

        num_samples = data['num_samples']

        for itr in range(10000):
            for idx, _ in enumerate(trans_tups):
                s = trans_tups[idx][0]
                a = trans_tups[idx][1]
                r = trans_tups[idx][2]
                ns = trans_tups[idx][3]
                sa = self.mdp.get_sa_encoding(s, a)
                exp_nsa = 0
                for na in range(self.mdp.n_action):
                    nsna = self.mdp.get_sa_encoding(ns, na)
                    if abs_q:
                        exp_nsa += self.abs_pie.get_prob(ns, na) * q_func[nsna]
                        #exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
                    else:
                        exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
                #if (idx + 1) % 100 == 0:
                #    q_func[sa] = q_func[sa] + alpha * (r - q_func[sa])
                #else:
                q_func[sa] = q_func[sa] + alpha * (r + self.gamma * exp_nsa - q_func[sa])
            if ((itr + 1) % 50 == 0):
                c = 0
                for i in range(q_dims):
                    if abs(q_func[i] - prev_q_func[i]) <= 1e-4:
                        c += 1
                if c == q_dims:
                    #print (q_func)
                    print ('converged, done, itr {}'.format(itr + 1))
                    break
            prev_q_func = copy.deepcopy(q_func)

            if ((itr + 1) % 100) == 0:
                print ('itr {} {}'.format(itr + 1, np.linalg.norm(q_func)))
            #    #print (q_func)

            #if (itr + 1) % 200 == 0:
            #    alpha = alpha / 2

        return q_func

class DiscreteDensityRatioEstimation2:
    def __init__(self, method, mdp, pie, gamma, abs_pie = None):
        self.gamma = gamma
        self.method = method
        self.mdp = mdp
        self.pie = pie
        self.abs_pie = abs_pie

    def compute(self, data, eps = 1e-10, alpha = 0.5, print_log = False):
        abs_w = True if (self.method == 'abs_ori' or self.method == 'abs_new') else False
        abs_q = True if (self.method == 'abs_new') else False

        # TODO use abs_w
        if abs_w:
            w_dims = self.mdp.n_abs_sa
        else:
            w_dims = self.mdp.n_sa

        # if abs_q will have passed abstract pie accordingly
        if abs_q:
            trans_tups = data['abs_trans']
            q_dims = self.mdp.n_abs_sa
        else:
            trans_tups = data['ground_trans']
            q_dims = self.mdp.n_sa
        prev_q_func = np.zeros((q_dims))
        q_func = np.zeros((q_dims))
        w_func = np.zeros((w_dims)) 
        num_samples = data['num_samples']

        for itr in range(10000):
            for idx, _ in enumerate(trans_tups):
                s = trans_tups[idx][0]
                a = trans_tups[idx][1]
                r = trans_tups[idx][2]
                ns = trans_tups[idx][3]
                sa = self.mdp.get_sa_encoding(s, a)
                exp_nsa = 0
                for na in range(self.mdp.n_action):
                    nsna = self.mdp.get_sa_encoding(ns, na)
                    if abs_q:
                        exp_nsa += self.abs_pie.get_prob(ns, na) * q_func[nsna]
                        #exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
                    else:
                        exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
                if (idx + 1) % 200 == 0:
                    q_func[sa] = q_func[sa] + alpha * (r - q_func[sa])
                else:
                    q_func[sa] = q_func[sa] + alpha * (r + self.gamma * exp_nsa - q_func[sa])
            if ((itr + 1) % 50 == 0):
                c = 0
                for i in range(q_dims):
                    if abs(q_func[i] - prev_q_func[i]) <= 1e-4:
                        c += 1
                if c == q_dims:
                    print (q_func)
                    print ('converged, done, itr {}'.format(itr + 1))
                    break
            prev_q_func = copy.deepcopy(q_func)

            if ((itr + 1) % 100) == 0:
                print (np.linalg.norm(q_func))
                print ('itr {}'.format(itr + 1))
                #print (q_func)

            #if (itr + 1) % 200 == 0:
            #    alpha = alpha / 2

        exp_init_val = 0
        densities = data['init_states']
        for d in densities:
            for a in range(self.mdp.n_action):
                if abs_q:
                    da = self.mdp.get_sa_encoding(self.mdp.phi(d), a)
                else:
                    da = self.mdp.get_sa_encoding(d, a)
                exp_init_val += densities[d] * self.pie.get_prob(d, a) * q_func[da]
        w_vals = {}
        for idx, _ in enumerate(trans_tups):
            s = trans_tups[idx][0]
            a = trans_tups[idx][1]
            r = trans_tups[idx][2]
            ns = trans_tups[idx][3]
            sa = self.mdp.get_sa_encoding(s, a)
            exp_nsa = 0
            for na in range(self.mdp.n_action):
                nsna = self.mdp.get_sa_encoding(ns, na)
                if abs_q:
                    exp_nsa += self.abs_pie.get_prob(ns, na) * q_func[nsna]
                    #exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
                else:
                    exp_nsa += self.pie.get_prob(ns, na) * q_func[nsna]
            if abs_q: 
                abs_sa = self.mdp.get_sa_encoding(s, a)
            else:
                abs_sa = self.mdp.get_sa_encoding(self.mdp.phi(s), a)
            if abs_sa not in w_vals:
                w_vals[abs_sa] = []

            #val = ((1. - self.gamma) * exp_init_val) / (q_func[sa] - self.gamma * exp_nsa)
            val = q_func[sa] - self.gamma * exp_nsa
            w_vals[abs_sa].append(val)

        for idx in range(w_dims):
            w_func[idx] = 0
            if idx in w_vals:
                 w_func[idx] = ((1. - self.gamma) * exp_init_val) / np.mean(w_vals[idx]) 
        self.w = w_func
        pdb.set_trace()
        return self.w 

    def get_W(self):
        return self.w

class TabularDensityRatioEstimationGAN:
    def __init__(self,
        seed,
        state_dims,
        action_dims,
        gamma,
        abs_state_dims = None,
        w_hidden_dim = -1,
        w_hidden_layers = 0,
        q_hidden_dim = -1,
        q_hidden_layers = 0,
        activation = None,
        W_lr = 1e-5,
        Q_lr = 1e-5,
        lam_lr = 0,
        unit_mean = True,
        alpha_w = False,
        method = None,
        w_reg = 0,
        q_reg = 0,
        alpha_r = False,
        uh_stabilizer = False):

        assert (method is not None)
        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.abs_state_dims = abs_state_dims
        self.gamma = gamma
        self.q_hidden_dim = q_hidden_dim
        self.q_hidden_layers = q_hidden_layers
        self.w_hidden_dim = w_hidden_dim
        self.w_hidden_layers = w_hidden_layers
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
        self.W_lr = W_lr
        self.Q_lr = Q_lr
        self.lam_lr = lam_lr
        self.unit_mean = unit_mean
        self.lam = torch.nn.Parameter(torch.tensor(0.), requires_grad = True)
        self.alpha_w = alpha_w
        self.alpha_r = alpha_r
        self.method = method
        self.w_reg = w_reg
        self.q_reg = q_reg
        self.uh_stabilizer = uh_stabilizer

        # default is ground dimensions
        w_state_input_dim = state_dims
        q_state_input_dim = state_dims # works for abs-pract-g-q
        if 'abs' in self.method:
            w_state_input_dim = abs_state_dims
        if self.method == 'abs' or self.method == 'abs-pract-abs-q':
            q_state_input_dim = abs_state_dims

        self.Q = NeuralNetwork(input_dims = q_state_input_dim * action_dims,
                                output_dims = 1,
                                hidden_dim = q_hidden_dim,
                                hidden_layers = q_hidden_layers,
                                activation = self.activation)

        self.W = NeuralNetwork(input_dims = w_state_input_dim * 1,#action_dims,
                                output_dims = 1,
                                hidden_dim = w_hidden_dim,
                                hidden_layers = w_hidden_layers,
                                activation = self.activation,
                                positivity_constraint = True)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.Q_lr)
        self.W_optimizer = torch.optim.Adam(self.W.parameters(), lr = self.W_lr)
        self.lam_optimizer = torch.optim.Adam([self.lam], lr = self.lam_lr)

    def train(self, data, epochs = 2000, print_log = False):
        abs_w = True if 'abs' in self.method else False

        if abs_w:
            # default assumes pure abstract
            w_inputs = data['abs_state_b']#data['abs_state_b_act_b']
            q_curr_inputs = data['abs_state_b_act_b']
            q_next_state_inputs = data['exp_abs_next_state_b_act_e']
            q_init_state_inputs = data['abs_init_state_act_e']
            if 'pract-abs-q' in self.method:
                # biased one
                q_next_state_inputs = data['exp_abs_next_state_b_act_e_pract']
            elif 'pract-g-q' in self.method:
                q_curr_inputs = data['state_b_act_b']
                q_next_state_inputs = data['exp_next_state_b_act_e']
                q_init_state_inputs = data['init_state_act_e'] 
        elif self.method == 'ground': 
            w_inputs = data['state_b']#data['state_b_act_b']
            #w_inputs = data['state_b_act_b']
            q_curr_inputs = data['state_b_act_b']
            q_next_state_inputs = data['exp_next_state_b_act_e']
            q_init_state_inputs = data['init_state_act_e'] 

        num_samples = data['num_samples']
        rewards = torch.from_numpy(data['rewards']).float()
        rewards = torch.reshape(rewards, (len(rewards), 1))

        mini_batch_size = min(2048, num_samples)
        min_obj = float('inf')
        best_epoch = -1

        for epoch in range(epochs):

            # first three have to be sync since they are (s, a, s', a') tuples
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            # abstraction should have already been done in format_data if applicable
            w_inputs_sub = w_inputs[subsamples]
            rewards_sub = rewards[subsamples]
            q_curr_inputs_sub = q_curr_inputs[subsamples]
            
            q_next_inputs_sub = q_next_state_inputs[subsamples] # either abs or ground

            q_init_inputs_sub = q_init_state_inputs # either abs or ground

            def _orth_reg(model):
                with torch.enable_grad():
                    orth_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            param_flat = param.view(param.shape[0], -1)
                            sym = torch.mm(param_flat, torch.t(param_flat))
                            sym -= torch.eye(param_flat.shape[0])
                            orth_loss = orth_loss + sym.abs().sum()
                    return orth_loss

            def _l2_reg(model):
                with torch.enable_grad():
                    l2_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(param, 2)))
                return l2_loss

            def _obj():
                w_outputs = self.W.forward(w_inputs_sub)
                w_norm = 1#torch.mean(w_outputs)
                w_norm_outputs = w_outputs / w_norm
                q_curr_outputs = self.Q.forward(q_curr_inputs_sub)
                q_next_outputs = self.Q.forward(q_next_inputs_sub)
                q_init_outputs = self.Q.forward(q_init_inputs_sub)
                td_error = 0
                if self.alpha_r:
                    td_error += rewards_sub
                td_error += self.gamma * q_next_outputs - q_curr_outputs


                obj = torch.mean(w_norm_outputs * td_error) + (1. - self.gamma) * torch.mean(q_init_outputs)
                if self.alpha_w:
                    obj += torch.mean(-torch.square(w_norm_outputs) / 2.) # using (x^2) / 2
                if self.unit_mean:
                    # lagrangian multiplier for equaling one
                    obj += self.lam * (1. - torch.mean(w_norm_outputs))
                if self.uh_stabilizer:
                    obj += -self.lam_lr * (torch.mean(torch.square(td_error)))
                if (epoch + 1) % 100 == 0 and False:#print_log:
                    #print ('q curr out {}'.format(torch.mean(q_curr_outputs)))
                    #print ('q next out {}'.format(torch.mean(q_next_outputs)))
                    print ('td error out {}'.format(torch.mean(td_error)))
                    print ('lam * (1 - w) {}'.format(self.lam * (1 - torch.mean(w_norm_outputs))))
                # adding negative to w since w maximizes obj
                #reg = -self.w_reg * _orth_reg(self.W) + self.q_reg * _orth_reg(self.Q)
                reg = -self.w_reg * _l2_reg(self.W) + self.q_reg * _l2_reg(self.Q)
                total_obj = obj + reg
                objs = {
                    'obj': obj,
                    'reg': reg,
                    'total_obj': total_obj
                }
                return objs 
            objs = _obj()
            total_obj = objs['total_obj']

            # clear gradients
            self.Q_optimizer.zero_grad()
            self.lam_optimizer.zero_grad()
            self.W_optimizer.zero_grad()
            
            # compute gradients
            total_obj.backward()

            # processing gradients
            nn.utils.clip_grad_value_(self.W.parameters(), clip_value = 1.0)
            nn.utils.clip_grad_value_(self.Q.parameters(), clip_value = 1.0)
            nn.utils.clip_grad_value_([self.lam], clip_value = 1.0)
            # negate gradients, for gradient ascent
            #for p in self.W.parameters():
            for p in self.Q.parameters():
                p.grad *= -1

            # gradient step
            self.Q_optimizer.step()
            self.lam_optimizer.step()
            self.W_optimizer.step()

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            if (epoch + 1) % 1000 == 0 or epoch == epochs - 1:
                print ('epoch {}'.format(epoch + 1))
                q_curr_outputs = self.Q.forward(q_curr_inputs_sub)
                q_next_outputs = self.Q.forward(q_next_inputs_sub)
                q_init_outputs = self.Q.forward(q_init_inputs_sub)
                #pdb.set_trace()
                mean_td = torch.mean(self.gamma * q_next_outputs - q_curr_outputs)
                dice = estimators.Dice(self.W)
                if epoch == epochs - 1:
                    pdb.set_trace()
                est = dice.estimate(w_inputs, data['rewards'], data['gammas'])
                print ('loss {} r est: {}, lam {}, qinit {}, mean td {}'.format([objs[name].item() for name in objs.keys()], est, self.lam, torch.mean(q_init_outputs), mean_td))

            if True:#abs(total_obj) < min_obj:
                min_obj = abs(total_obj)
                self.best_W = copy.deepcopy(self.W)
                best_epoch = epoch

        dice = estimators.Dice(self.best_W)
        est = dice.estimate(w_inputs, data['rewards'], data['gammas'])
        print ('best epoch {}, obj {}, r est {}'.format(best_epoch, min_obj, est))

    def get_W(self):
        return self.best_W#self.W


