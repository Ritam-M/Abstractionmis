import torch
from torch import nn
import numpy as np
from policies import NeuralNetwork, RBFKernel
import copy
import time
import estimators
import pdb

class ContinuousDensityRatioEstimationGAN:
    def __init__(self,
        seed,
        state_dims,
        action_dims,
        gamma,
        pie,
        abs_state_dims = None,
        w_hidden_dim = 32,
        w_hidden_layers = 1,
        q_hidden_dim = 32,
        q_hidden_layers = 1,
        activation = 'relu',
        W_lr = 1e-5,
        Q_lr = 1e-5,
        lam_lr = 1e-4,
        unit_norm = True,
        method = None,
        w_reg = 0,
        q_reg = 0,
        alpha_w = False,
        alpha_r = False,
        rank_penalty = False,
        uh_stabilizer = False,
        w_pos = True):

        assert (method is not None)
        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.abs_state_dims = abs_state_dims
        self.pie = pie
        self.gamma = gamma
        self.q_hidden_dim = q_hidden_dim
        self.q_hidden_layers = q_hidden_layers
        self.w_hidden_dim = w_hidden_dim
        self.w_hidden_layers = w_hidden_layers
        self.activation = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.W_lr = W_lr
        self.Q_lr = Q_lr
        self.lam_lr = lam_lr
        self.unit_norm = unit_norm
        self.lam = torch.nn.Parameter(torch.tensor(0.), requires_grad = True)
        self.alpha_w = alpha_w
        self.alpha_r = alpha_r
        self.method = method
        self.w_reg = w_reg
        self.q_reg = q_reg
        self.rank_penalty = rank_penalty
        self.uh_stabilizer = uh_stabilizer

        w_state_input_dim = state_dims
        q_state_input_dim = state_dims
        if self.method == 'abs_ori' or self.method == 'abs_new':
            w_state_input_dim = abs_state_dims
        if self.method == 'abs_new':
            q_state_input_dim = abs_state_dims

        self.Q = NeuralNetwork(input_dims = q_state_input_dim + action_dims,
                                output_dims = 1,
                                hidden_dim = q_hidden_dim,
                                hidden_layers = q_hidden_layers,
                                activation = self.activation)

        self.W = NeuralNetwork(input_dims = w_state_input_dim + action_dims,
                                output_dims = 1,
                                hidden_dim = w_hidden_dim,
                                hidden_layers = w_hidden_layers,
                                activation = self.activation,
                                positivity_constraint = w_pos)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.Q_lr)
        self.W_optimizer = torch.optim.Adam(self.W.parameters(), lr = self.W_lr)
        self.lam_optimizer = torch.optim.Adam([self.lam], lr = self.lam_lr)

    def train(self, data, epochs = 2000, print_log = True):
        abs_w = True if (self.method == 'abs_ori' or self.method == 'abs_new') else False
        abs_q = True if (self.method == 'abs_new') else False

        if abs_w:
            w_inputs = data['abs_state_b_act_b']
            #w_inputs = data['abs_state_b']#data['abs_state_b_act_b']
        else:
            #w_inputs = data['state_b']#data['state_b_act_b']
            w_inputs = data['state_b_act_b']

        if abs_q:
            q_curr_inputs = data['abs_state_b_act_b']
            q_next_state_inputs = data['abs_next_state_b']
            q_init_state_inputs = data['abs_init_state']
        else:
            q_curr_inputs = data['state_b_act_b']
            q_next_state_inputs = data['next_state_b']
            q_init_state_inputs = data['init_state']

        num_samples = data['num_samples']
        rewards = torch.from_numpy(data['rewards']).float()
        rewards = torch.reshape(rewards, (len(rewards), 1))

        mini_batch_size = min(2048, num_samples)
        min_obj = float('inf')
        best_epoch = -1

        r_ests = []
        q_ranks = []
        w_ranks = []

        for epoch in range(epochs):

            # first three have to be sync since they are (s, a, s', a') tuples
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            # abstraction should have already been done in format_data if applicable
            w_inputs_sub = w_inputs[subsamples]
            rewards_sub = rewards[subsamples]
            q_curr_inputs_sub = q_curr_inputs[subsamples]
            
            next_ground_states = data['next_state_b'][subsamples] # always ground
            q_next_state_inputs_sub = q_next_state_inputs[subsamples] # either abs or ground
            q_next_inputs_sub = np.concatenate((q_next_state_inputs_sub, self.pie.batch_sample(next_ground_states)), axis = 1) # sample from ground, append to abs/ground

            # the init state is independent of the the above transitions
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            init_ground_states = data['init_state'][subsamples]
            q_init_state_inputs_sub = q_init_state_inputs[subsamples] # either abs or ground
            q_init_inputs_sub = np.concatenate((q_init_state_inputs_sub, self.pie.batch_sample(init_ground_states)), axis = 1)

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

                rank_reg = 0
                if self.rank_penalty:
                    pen_w_curr = self.W.get_penultimate(w_inputs_sub)
                    #pen_q_curr = self.Q.get_penultimate(q_curr_inputs_sub)
                    #pen_q_next = self.Q.get_penultimate(q_next_inputs_sub)
                    #feat_prod = torch.sum(pen_q_curr * pen_q_next, dim = -1)
                    #assert len(feat_prod) == mini_batch_size
                    #rank_reg = torch.sum(feat_prod)
                    _, pen_w_sing, _  = torch.linalg.svd(pen_w_curr)
                    rank_reg = -(torch.square(torch.max(pen_w_sing)) - torch.square(torch.min(pen_w_sing)))

                td_error = 0
                if self.alpha_r:
                    td_error += rewards_sub
                td_error += self.gamma * q_next_outputs - q_curr_outputs

                obj = torch.mean(w_norm_outputs * td_error) + (1. - self.gamma) * torch.mean(q_init_outputs)
                if self.alpha_w:
                    obj += torch.mean(-torch.square(w_norm_outputs) / 2.) # using (x^2) / 2
                if self.unit_norm:
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
                #obj = torch.square(obj)
                total_obj = obj + reg + 0.001 * rank_reg
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
            for p in self.W.parameters():
            #for p in self.Q.parameters():
                p.grad *= -1
            
            # gradient step
            self.Q_optimizer.step()
            self.lam_optimizer.step()
            self.W_optimizer.step()

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            if (epoch + 1) % 1000 == 0 or epoch == epochs - 1:
                subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
                w_inputs_sub = w_inputs[subsamples]
                q_curr_inputs_sub = q_curr_inputs[subsamples]
                pen_w = self.W.get_penultimate(w_inputs_sub).detach().numpy()
                pen_q = self.Q.get_penultimate(q_curr_inputs_sub).detach().numpy()

                th = 1 - 0.01
                _, s, _ = np.linalg.svd(pen_q)
                den = np.sum(s)
                cum_sum = 0
                for i in range(len(s)):
                    cum_sum += s[i] 
                    rat = cum_sum / den
                    if rat >= th:
                        break

                _, s, _ = np.linalg.svd(pen_w)
                den = np.sum(s)
                cum_sum = 0
                for j in range(len(s)):
                    cum_sum += s[j] 
                    rat = cum_sum / den
                    if rat >= th:
                        break
                w_rank = j#np.linalg.matrix_rank(pen_w)
                q_rank = i#np.linalg.matrix_rank(pen_q)
                w_ranks.append(w_rank)
                q_ranks.append(q_rank)
                dice = estimators.Dice(self.W)
                est = dice.estimate(w_inputs, data['rewards'], data['gammas'])
                r_ests.append(est)
                if print_log:
                    print ('epoch {}'.format(epoch + 1))
                    print ('loss {} r est: {}, lam {}, w rank {}, q rank {}'.format([objs[name].item() for name in objs.keys()], est, self.lam, w_rank, q_rank))

            if True:#abs(total_obj) < min_obj:
                min_obj = abs(total_obj)
                self.best_W = self.W
                #self.best_W = copy.deepcopy(self.W)
                best_epoch = epoch

        dice = estimators.Dice(self.best_W)
        est = dice.estimate(w_inputs, data['rewards'], data['gammas'])
        print ('best epoch {}, obj {}, r est {}'.format(best_epoch, min_obj, est))

        self.r_ests = np.array(r_ests)
        self.q_ranks = np.array(q_ranks)
        self.w_ranks = np.array(w_ranks)

    def get_metrics(self):
        metrics = {
            'r_ests': self.r_ests,
            'q_ranks': self.q_ranks,
            'w_ranks': self.w_ranks
        }
        return metrics

    def get_W(self):
        return self.best_W#self.W

''' unused code below '''

class ContinuousDensityRatioEstimationKernel:
    def __init__(self,
        seed,
        state_dims,
        action_dims,
        gamma,
        pie,
        abs_state_dims = None,
        hidden_dim = 32,
        hidden_layers = 1,
        activation = 'relu',
        W_lr = 1e-4,
        rbf_sigma_scale = 1.,
        w_reg = 5e-4,
        method = None):

        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.abs_state_dims = abs_state_dims
        self.phi = None
        self.pie = pie
        self.gamma = gamma
        self.rbf_sigma_scale = rbf_sigma_scale
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.W_lr = W_lr
        self.w_reg = w_reg
        self.method = method

        w_state_input_dim = state_dims
        if self.method == 'abs_ori' or self.method == 'abs_new':
            w_state_input_dim = abs_state_dims

        self.W = NeuralNetwork(input_dims = w_state_input_dim + action_dims,
                                output_dims = 1,
                                hidden_dim = hidden_dim,
                                hidden_layers = hidden_layers,
                                activation = self.activation,
                                positivity_constraint = True)

        self.W_optimizer = torch.optim.Adam(self.W.parameters(), lr = self.W_lr)

    def train(self, data, epochs = 3000, print_log = False):
        abs_w = True if (self.method == 'abs_ori' or self.method == 'abs_new') else False
        abs_q = True if (self.method == 'abs_new') else False

        if abs_w:
            w_inputs = data['abs_state_b_act_b']
        else:
            w_inputs = data['state_b_act_b']

        if abs_q:
            q_curr_inputs = data['abs_state_b_act_b']
            q_next_state_inputs = data['abs_next_state_b']
            q_init_state_inputs = data['abs_init_state']
        else:
            q_curr_inputs = data['state_b_act_b']
            q_next_state_inputs = data['next_state_b']
            q_init_state_inputs = data['init_state']

        num_samples = data['num_samples']

        subsamples = np.random.choice(len(q_curr_inputs), 1000)
        med_q_curr_inputs_sub = q_curr_inputs[subsamples]
        med_dist = np.median(np.sqrt(np.sum(np.square(med_q_curr_inputs_sub[None, :, :] - med_q_curr_inputs_sub[:, None, :]), axis = -1)))
        Kxx = RBFKernel(sigma = med_dist / self.rbf_sigma_scale)

        mini_batch_size = min(2048, num_samples)
        min_obj = float('inf')
        best_epoch = -1 

        for epoch in range(epochs):
            z_norm = torch.mean(self.W.forward(w_inputs))
            
            # first three have to be sync since they are (s, a, s', a') tuples
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            # abstraction should have already been done in format_data if applicable
            w_inputs_sub1 = w_inputs[subsamples]
            q_curr_inputs_sub1 = q_curr_inputs[subsamples]
           
            # TODO make separate method since repeated functionality 
            next_ground_states = data['next_state_b'][subsamples] # always ground
            q_next_state_inputs_sub = q_next_state_inputs[subsamples] # either abs or ground
            q_next_inputs_sub1 = np.concatenate((q_next_state_inputs_sub,\
                                    self.pie.batch_sample(next_ground_states)), axis = 1) # sample from ground, append to abs/ground

            # TODO skipping double sampling for now
            w_inputs_sub2 = w_inputs_sub1
            q_curr_inputs_sub2 = q_curr_inputs_sub1
            q_next_inputs_sub2 = q_next_inputs_sub1

            # the init state is independent of the the above transitions
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            init_ground_states = data['init_state'][subsamples]
            q_init_state_inputs_sub = q_init_state_inputs[subsamples] # either abs or ground
            q_init_inputs_sub1 = np.concatenate((q_init_state_inputs_sub,\
                                    self.pie.batch_sample(init_ground_states)), axis = 1)
            q_init_inputs_sub2 = q_init_inputs_sub1

            def _l2_reg(model):
                with torch.enable_grad():
                    l2_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(param, 2)))
                return l2_loss

            def _W_loss_1():
                # Appendix 2, Lemma 17 Uhera
                w_outputs_sub1 = self.W.forward(w_inputs_sub1)
                w_outputs_sub2 = self.W.forward(w_inputs_sub2)
                first = (self.gamma ** 2) * torch.mean( \
                    w_outputs_sub1\
                    * w_outputs_sub2\
                    * torch.reshape(Kxx(q_next_inputs_sub1, q_next_inputs_sub2), (-1, 1))) / (z_norm * z_norm)
                second = torch.mean(w_outputs_sub1\
                    * w_outputs_sub2\
                    * torch.reshape(Kxx(q_curr_inputs_sub1, q_curr_inputs_sub2), (-1, 1))) / (z_norm * z_norm)
                third =  ((1. - self.gamma) ** 2) * torch.mean(torch.reshape(Kxx(q_init_inputs_sub1, q_init_inputs_sub2), (-1, 1)))
                fourth = -2 * self.gamma\
                    * torch.mean(w_outputs_sub1\
                    * w_outputs_sub2\
                    * torch.reshape(Kxx(q_next_inputs_sub1, q_curr_inputs_sub2), (-1, 1))) / (z_norm * z_norm)
                fifth = 2 * self.gamma * (1 - self.gamma) * self.gamma\
                    * torch.mean(w_outputs_sub1\
                    * torch.reshape(Kxx(q_next_inputs_sub1, q_init_inputs_sub1), (-1, 1))) / (z_norm * z_norm)
                sixth = -2 * (1 - self.gamma)\
                    * torch.mean(w_outputs_sub1\
                    * self.W.forward(q_init_inputs_sub1)
                    * torch.reshape(Kxx(q_curr_inputs_sub1, q_init_inputs_sub1), (-1, 1))) / (z_norm * z_norm)
                obj = first + second + third + fourth + fifth + sixth
                reg = self.w_reg * _l2_reg(self.W)
                
                total_obj = obj + reg
                objs = {
                    'obj': obj,
                    'reg': reg,
                    'total_obj': total_obj
                }
                return objs

            def _W_loss_2():
                # 1st
                loss = (self.gamma ** 2)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)\
                        * (self.W.forward(w_inputs_sub2, requires_grad = True) / z_norm)\
                        * Kxx(next_trans1, next_trans2))
                # 2nd
                loss += (self.gamma ** 2)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)\
                        * (self.W.forward(w_inputs_sub2, requires_grad = True) / z_norm)\
                        * Kxx(curr_trans1, curr_trans2))

                # 3rd
                if self.phi is None:
                    loss += ((1 - self.gamma) ** 2)\
                            * torch.mean((self.W.forward(init_trans1_b, requires_grad = True) / z_norm)\
                            * (self.W.forward(init_trans2_b, requires_grad = True) / z_norm)\
                            * Kxx(init_trans1_b, init_trans2_b))
                else:
                    loss += ((1 - self.gamma) ** 2)\
                            * torch.mean((self.W.forward(abs_init_trans1_b, requires_grad = True) / z_norm)
                            * (self.W.forward(abs_init_trans2_b, requires_grad = True) / z_norm)
                            * Kxx(init_trans1_b, init_trans2_b))
                # 4th
                loss += ((1 - self.gamma) ** 2)\
                        * torch.mean(Kxx(init_trans1_e, init_trans2_e))

                # 5th
                loss += (-2 * (self.gamma ** 2))\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                        * (self.W.forward(w_inputs_sub2, requires_grad = True) / z_norm)
                        * Kxx(next_trans1, curr_trans2))

                # 6th
                if self.phi is None:
                    loss += -2 * self.gamma * (1 - self.gamma)\
                            * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                            * (self.W.forward(init_trans1_b, requires_grad = True) / z_norm)
                            * Kxx(next_trans1, init_trans1_b))
                else:
                    loss += -2 * self.gamma * (1 - self.gamma)\
                            * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                            * (self.W.forward(abs_init_trans1_b, requires_grad = True) / z_norm)
                            * Kxx(next_trans1, init_trans1_b))

                # 7th
                loss += 2 * self.gamma * (1. - self.gamma)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                        * Kxx(next_trans1, init_trans1_e))
                # 8th
                if self.phi is None:
                    loss += 2 * self.gamma * (1. - self.gamma)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                        * (self.W.forward(init_trans1_b, requires_grad = True) / z_norm)
                        * Kxx(curr_trans1, init_trans1_b))
                else:
                    loss += 2 * self.gamma * (1. - self.gamma)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                        * (self.W.forward(abs_init_trans1_b, requires_grad = True) / z_norm)
                        * Kxx(curr_trans1, init_trans1_b))

                # 9th
                loss += -2 * self.gamma * (1. - self.gamma)\
                        * torch.mean((self.W.forward(w_inputs_sub1, requires_grad = True) / z_norm)
                        * Kxx(curr_trans1, init_trans1_e))
                
                # 10th
                if self.phi is None:
                    loss += -2 * (1. - self.gamma) ** 2\
                        * torch.mean((self.W.forward(init_trans1_b, requires_grad = True) / z_norm)
                        * Kxx(init_trans1_b, init_trans1_e))
                else:
                    loss += -2 * (1. - self.gamma) ** 2\
                        * torch.mean((self.W.forward(abs_init_trans1_b, requires_grad = True) / z_norm)
                        * Kxx(init_trans1_b, init_trans1_e))
                return loss

            objs = _W_loss_1()
            total_obj = objs['total_obj']

            # minimze loss
            self.W_optimizer.zero_grad()
            total_obj.backward()
            
            # processing gradients
            nn.utils.clip_grad_value_(self.W.parameters(), clip_value = 1.0)
            self.W_optimizer.step()

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            if abs(total_obj) < min_obj:
                min_obj = abs(total_obj)
                self.best_W = copy.deepcopy(self.W)
                best_epoch = epoch

            if (epoch + 1) % 250 == 0 or epoch == epochs - 1:
                print ('epoch {}'.format(epoch + 1))
                dice = estimators.Dice(self.W)
                if abs_w:
                    est = dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'])
                else:
                    est = dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'])
                print ('loss {} r est: {}'.format([objs[name].item() for name in objs.keys()], est))

        dice = estimators.Dice(self.best_W)
        if abs_w:
            est = dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'])
        else:
            est = dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'])
        print ('best epoch {}, obj {}, r est {}'.format(best_epoch, min_obj, est))
        
    def get_W(self):
        return self.best_W

class ContinuousFQE:
    def __init__(self,
        seed,
        state_dims,
        action_dims,
        gamma,
        pie,
        abs_state_dims = None,
        q_hidden_dim = 32,
        q_hidden_layers = 1,
        activation = 'relu',
        Q_lr = 1e-5,
        method = None,
        q_reg = 0,
        rank_penalty = False):

        assert (method is not None)
        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.abs_state_dims = abs_state_dims
        self.pie = pie
        self.gamma = gamma
        self.q_hidden_dim = q_hidden_dim
        self.q_hidden_layers = q_hidden_layers
        self.activation = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.Q_lr = Q_lr
        self.method = method
        self.q_reg = q_reg
        self.rank_penalty = rank_penalty

        q_state_input_dim = state_dims
        if self.method == 'abs':
            q_state_input_dim = abs_state_dims

        self.Q = NeuralNetwork(input_dims = q_state_input_dim + action_dims,
                                output_dims = 1,
                                hidden_dim = q_hidden_dim,
                                hidden_layers = q_hidden_layers,
                                activation = self.activation)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.Q_lr)

    def train(self, data, epochs = 2000, print_log = True):
        abs_q = True if (self.method == 'abs') else False

        if abs_q:
            q_curr_inputs = data['abs_state_b_act_b']
            q_next_state_inputs = data['abs_next_state_b']
            q_init_state_inputs = data['abs_init_state']
        else:
            q_curr_inputs = data['state_b_act_b']
            q_next_state_inputs = data['next_state_b']
            q_init_state_inputs = data['init_state']

        num_samples = data['num_samples']
        rewards = torch.from_numpy(data['rewards']).float()
        rewards = torch.reshape(rewards, (len(rewards), 1))

        mini_batch_size = min(2048, num_samples)
        min_obj = float('inf')
        best_epoch = -1

        r_ests = []
        q_ranks = []

        for epoch in range(epochs):

            # first three have to be sync since they are (s, a, s', a') tuples
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            # abstraction should have already been done in format_data if applicable
            rewards_sub = rewards[subsamples]
            q_curr_inputs_sub = q_curr_inputs[subsamples]
            
            next_ground_states = data['next_state_b'][subsamples] # always ground
            q_next_state_inputs_sub = q_next_state_inputs[subsamples] # either abs or ground
            q_next_inputs_sub = np.concatenate((q_next_state_inputs_sub, self.pie.batch_sample(next_ground_states)), axis = 1) # sample from ground, append to abs/ground

            # the init state is independent of the the above transitions
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            init_ground_states = data['init_state'][subsamples]
            q_init_state_inputs_sub = q_init_state_inputs[subsamples] # either abs or ground
            q_init_inputs_sub = np.concatenate((q_init_state_inputs_sub, self.pie.batch_sample(init_ground_states)), axis = 1)

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
                q_curr_outputs = self.Q.forward(q_curr_inputs_sub)
                q_next_outputs = self.Q.forward(q_next_inputs_sub)
                q_init_outputs = self.Q.forward(q_init_inputs_sub)

                rank_reg = 0
                if self.rank_penalty:
                    pen_w_curr = self.W.get_penultimate(w_inputs_sub)
                    #pen_q_curr = self.Q.get_penultimate(q_curr_inputs_sub)
                    #pen_q_next = self.Q.get_penultimate(q_next_inputs_sub)
                    #feat_prod = torch.sum(pen_q_curr * pen_q_next, dim = -1)
                    #assert len(feat_prod) == mini_batch_size
                    #rank_reg = torch.sum(feat_prod)
                    _, pen_w_sing, _  = torch.linalg.svd(pen_w_curr)
                    rank_reg = -(torch.square(torch.max(pen_w_sing)) - torch.square(torch.min(pen_w_sing)))

                target = rewards_sub + self.gamma * q_next_outputs

                obj = torch.mean(torch.square(q_curr_outputs - target))
                if (epoch + 1) % 100 == 0 and False:#print_log:
                    #print ('q curr out {}'.format(torch.mean(q_curr_outputs)))
                    #print ('q next out {}'.format(torch.mean(q_next_outputs)))
                    print ('td error out {}'.format(torch.mean(td_error)))
                    print ('lam * (1 - w) {}'.format(self.lam * (1 - torch.mean(w_norm_outputs))))
                # adding negative to w since w maximizes obj
                #reg = -self.w_reg * _orth_reg(self.W) + self.q_reg * _orth_reg(self.Q)
                reg = self.q_reg * _l2_reg(self.Q)
                #obj = torch.square(obj)
                total_obj = obj + reg + 0.001 * rank_reg
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
            
            # compute gradients
            total_obj.backward()

            # processing gradients
            nn.utils.clip_grad_value_(self.Q.parameters(), clip_value = 1.0)
            
            # gradient step
            self.Q_optimizer.step()

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            if (epoch + 1) % 1000 == 0 or epoch == epochs - 1:
                subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
                q_curr_inputs_sub = q_curr_inputs[subsamples]
                pen_q = self.Q.get_penultimate(q_curr_inputs_sub).detach().numpy()

                th = 1 - 0.01
                _, s, _ = np.linalg.svd(pen_q)
                den = np.sum(s)
                cum_sum = 0
                for i in range(len(s)):
                    cum_sum += s[i] 
                    rat = cum_sum / den
                    if rat >= th:
                        break

                q_rank = i#np.linalg.matrix_rank(pen_q)
                q_ranks.append(q_rank)
                est = (1. - self.gamma) * torch.mean(self.Q.forward(q_init_inputs_sub))
                r_ests.append(est)
                if print_log:
                    print ('epoch {}'.format(epoch + 1))
                    print ('loss {} r est: {},  q rank {}'.format([objs[name].item() for name in objs.keys()], est, q_rank))

            if True:#abs(total_obj) < min_obj:
                min_obj = abs(total_obj)
                best_epoch = epoch

        #self.r_ests = np.array(r_ests)
        #self.q_ranks = np.array(q_ranks)

    def get_metrics(self):
        metrics = {
            'r_ests': self.r_ests,
            'q_ranks': self.q_ranks,
        }
        return metrics

    def get_Q(self):
        return self.Q


