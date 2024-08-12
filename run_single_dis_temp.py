from __future__ import print_function
from __future__ import division

import numpy as np
np.set_printoptions(suppress=True)
import pdb
import argparse
import random

from toymdp import GraphMDP
from policies import DiscreteGridworldElbow, DiscreteGridworldRandomElbow, AbstractDiscreteGridworld, DiscreteGridworldElbowSlow, ToyMDPPolicy
import estimators
import utils
from discrete_density_ratio import DiscreteDensityRatioEstimation, TabularDualDice

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# saving
parser.add_argument('--outfile', default = None)

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--stoch_prob', default = 0., type = float)
parser.add_argument('--mdp_num', default = 0, type = int)
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--pi_set', default = 0, type = int)
parser.add_argument('--mix_ratio', default = 0.7, type = float)
parser.add_argument('--oracle_batch_size', default = 5, type = int)
parser.add_argument('--epochs', default = 2000, type = int)

# variables
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--batch_size', default = 5, type = int)
parser.add_argument('--traj_len', default = 5, type = int)
parser.add_argument('--eps', default = 1e-10, type = float)
parser.add_argument('--Q_lr', default = 1e-4, type = float)
parser.add_argument('--W_lr', default = 1e-4, type = float)
parser.add_argument('--lam_lr', default = 1e-4, type = float)
parser.add_argument('--compute_true_ratios', type = str2bool, default = 'true')

# misc
parser.add_argument('--plot', default = 'false', type = str2bool)
parser.add_argument('--exp_name', default = 'ope', type = str)

FLAGS = parser.parse_args()

'''
Bisim and pi act equal: MDP #0 and pi set #0
no Bisim and pi act equal: MDP #1 and pi set #0
bisim and no pi act equal: MDP #2 and pi set #1
Not bisim and not pi act equal: MDP #3 and pi set #1
'''

def env_setup():
    if FLAGS.env_name == 'ToyMDP':
        if FLAGS.mdp_num == 0:
            phi = {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
            abstraction = {
                'phi': phi,
                'num_abs_states': 3
            }
            num_states = 4
            num_actions = 2
            rewards = np.zeros((num_states, num_actions))
            rewards[1, 0] = 1.
            rewards[2, 0] = 1.
            env = GraphMDP(num_states, num_actions, rewards = rewards, init_state = 0, abstraction = abstraction)
            
            # setup MDP
            env.set_transition_prob(0, 0, 1, 1.0)
            env.set_transition_prob(0, 1, 2, 1.0)
            env.set_transition_prob(1, 0, 3, 1.0)
            env.set_transition_prob(2, 0, 3, 1.0)
            env.set_transition_prob(3, 0, 0, 1.0)
        elif FLAGS.mdp_num == 1:
            phi = {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
            abstraction = {
                'phi': phi,
                'num_abs_states': 3
            }
            num_states = 4
            num_actions = 2
            rewards = np.zeros((num_states, num_actions))
            rewards[1, 0] = 1
            rewards[2, 0] = 1
            env = GraphMDP(num_states, num_actions, rewards = rewards, init_state = 0, abstraction = abstraction)
            
            # setup MDP
            env.set_transition_prob(0, 0, 1, 1.0)
            env.set_transition_prob(0, 1, 2, 1.0)
            env.set_transition_prob(1, 0, 3, 1.0)
            env.set_transition_prob(2, 0, 1, 1.0)
            env.set_transition_prob(3, 0, 0, 1.0)
        elif FLAGS.mdp_num == 2:
            phi = {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
            abstraction = {
                'phi': phi,
                'num_abs_states': 3
            }
            num_states = 4
            num_actions = 2
            rewards = np.zeros((num_states, num_actions))
            rewards[1, 0] = 1
            rewards[2, 0] = 1
            env = GraphMDP(num_states, num_actions, rewards = rewards, init_state = 0, abstraction = abstraction)
            
            # setup MDP
            env.set_transition_prob(0, 0, 1, 1.0)
            env.set_transition_prob(0, 1, 2, 1.0)
            env.set_transition_prob(1, 0, 3, 1.0)
            env.set_transition_prob(1, 1, 2, 1.0)
            env.set_transition_prob(2, 0, 3, 1.0)
            env.set_transition_prob(2, 1, 1, 1.0)
            env.set_transition_prob(3, 0, 0, 1.0)
        elif FLAGS.mdp_num == 3:
            phi = {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
            abstraction = {
                'phi': phi,
                'num_abs_states': 3
            }
            num_states = 4
            num_actions = 2
            rewards = np.zeros((num_states, num_actions))
            rewards[1, 0] = 1
            rewards[2, 0] = 1
            env = GraphMDP(num_states, num_actions, rewards = rewards, init_state = 0, abstraction = abstraction)
            
            # setup MDP
            env.set_transition_prob(0, 0, 1, 1.0)
            env.set_transition_prob(0, 1, 2, 1.0)
            env.set_transition_prob(1, 0, 3, 1.0)
            env.set_transition_prob(1, 1, 2, 1.0)
            env.set_transition_prob(2, 0, 1, 1.0)
            env.set_transition_prob(2, 1, 3, 1.0)
            env.set_transition_prob(3, 0, 0, 1.0)
    return env

def policies_setup(env):
    mix_ratios = [FLAGS.mix_ratio, 1. - FLAGS.mix_ratio]
    if FLAGS.env_name == 'ToyMDP':
        if FLAGS.pi_set == 0:
            num_states = env.n_state
            num_actions = env.n_action
            pie = np.zeros((num_states, num_actions))
            pie[0][0] = 0.01
            pie[0][1] = 0.99
            pie[1][0] = 1.
            pie[2][0] = 1.
            pie[3][0] = 1.
            pie = ToyMDPPolicy(num_actions, pie)

            pib = np.zeros((num_states, num_actions))
            pib[0][0] = 0.99
            pib[0][1] = 0.01
            pib[1][0] = 1.
            pib[2][0] = 1.
            pib[3][0] = 1.
            pib = ToyMDPPolicy(num_actions, pib)
        elif FLAGS.pi_set == 1:
            num_states = env.n_state
            num_actions = env.n_action
            pie = np.zeros((num_states, num_actions))
            pie[0][0] = 0.01
            pie[0][1] = 0.99
            pie[1][0] = 0.9
            pie[1][1] = 0.1
            pie[2][0] = 0.1
            pie[2][1] = 0.9
            pie[3][0] = 1.

            pie = ToyMDPPolicy(num_actions, pie)

            pib = np.zeros((num_states, num_actions))
            pib[0][0] = 0.99
            pib[0][1] = 0.01
            pib[1][0] = 0.1
            pib[1][1] = 0.9
            pib[2][0] = 0.9
            pib[2][1] = 0.1
            pib[3][0] = 1.
            pib = ToyMDPPolicy(num_actions, pib)
    return pie, pib

def _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie):
    np.random.seed(seed)
    initial_states = mdp.get_initial_state_dist()

    np.random.seed(seed)
    paths, f, rew, _ = utils.collect_data_discrete(mdp, pib, batch_size, truncated_horizon = truncated_horizon)

    data = {
        'initial_states': initial_states,
        'data': paths
    }
    # format data into relevant inputs needed by loss function
    np.random.seed(seed)
    data = utils.format_data_discrete(mdp, data, pie, gamma, abs_pie)
    return data

def _data_prep_temp(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie):
    np.random.seed(seed)
    initial_states = mdp.get_initial_state_dist()

    np.random.seed(seed)
    paths, f, rew, _ = utils.collect_data_discrete(mdp, pie, batch_size, truncated_horizon = truncated_horizon)

    data = {
        'initial_states': initial_states,
        'data': paths
    }
    # format data into relevant inputs needed by loss function
    np.random.seed(seed)
    data = utils.format_data_discrete(mdp, data, pie, gamma, abs_pie)
    return data

def run_experiment(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie, true_dens = None):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie)
    #data = _data_prep_temp(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie)

    # true densities
    np.random.seed(seed)
    true_g_est, true_a_est = true_off_estimate(seed, data, gamma, true_dens)

    np.random.seed(seed)
    #g_est_mwl = off_estimate(seed, data, gamma, method = 'ground')
    g_est_mwl, _ = off_estimate_dual(seed, data, gamma, mdp, pie, method = 'ground')
    
    np.random.seed(seed)
    abs_est_mwl = 0#off_estimate(seed, data, gamma, method = 'abs')
    
    np.random.seed(seed)
    #abs_est_mwl_pract = off_estimate(seed, data, gamma, method = 'abs-pract')
    abs_est_mwl_pract, _ = off_estimate_dual(seed, data, gamma, mdp, pie, method = 'abs-pract')
    
    results = (true_g_est, true_a_est, g_est_mwl, abs_est_mwl, abs_est_mwl_pract)
    print (results)
    return results

def run_experiment_densities(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie, true_dens = None):
    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie)

    abs_pie_true_dens = true_dens['abs_pie']

    np.random.seed(seed)
    est_pie_dens = off_dual_dens_estimate(seed, data, true_dens, gamma, mdp, pie, method = 'abs-pract')
    
    assert len(abs_pie_true_dens) == len(est_pie_dens)

    return (abs_pie_true_dens, est_pie_dens)

def on_policy_estimate(seed, batch_size, truncated_horizon, gamma, mdp, pie):
    # on-policy ground case
    g_pie_paths, _, _, _ = utils.collect_data_discrete(mdp, pie, batch_size, truncated_horizon)
    g_pie_estimator = estimators.OnPolicy(pie, gamma)
    g_pie_est = g_pie_estimator.estimate(g_pie_paths)
    print ('true: {}'.format(g_pie_est))
    return g_pie_est

def true_off_estimate(seed, data, gamma, true_dens):
    rews = data['rewards']
    gammas = data['gammas']
    ge_t = true_dens['g_pie'][np.argmax(data['state_b_act_b'], axis = 1)]
    gb_t = true_dens['g_pib'][np.argmax(data['state_b_act_b'], axis = 1)]
    gr = ge_t / gb_t
    d_pie = true_dens['g_pie'][np.unique(np.argmax(data['state_b_act_b'], axis = 1))]
    d_pib = true_dens['g_pib'][np.unique(np.argmax(data['state_b_act_b'], axis = 1))]
    x = d_pie / d_pib
    print ('gr pie densities {}'.format(d_pie))
    print ('gr pib densities {}'.format(d_pib))
    print ('true ground ratios {}'.format(x))
    g_est = np.mean(gr * rews)
    print ('est with true gr {}'.format(g_est))

    #pdb.set_trace()
    ae_t = true_dens['abs_pie'][np.argmax(data['abs_state_b_act_b'], axis = 1)]
    ab_t = true_dens['abs_pib'][np.argmax(data['abs_state_b_act_b'], axis = 1)]
    ar = ae_t / ab_t
    d_abs_pie = true_dens['abs_pie'][np.unique(np.argmax(data['abs_state_b_act_b'], axis = 1))]
    d_abs_pib = true_dens['abs_pib'][np.unique(np.argmax(data['abs_state_b_act_b'], axis = 1))]
    y = d_abs_pie / d_abs_pib
    #sub_ar = ar[np.unique(np.argmax(data['abs_state_b_act_b'], axis = 1))]
    #print (sub_ar)
    print ('abs pie densities {}'.format(d_abs_pie))
    print ('abs pib densities {}'.format(d_abs_pib))
    print ('true abs ratios {}'.format(y))
    a_est = np.mean(ar * rews)
    print ('est with true abs {}'.format(a_est))

    return g_est, a_est

def off_estimate(seed, data, gamma, method):
    d = DiscreteDensityRatioEstimation(method, gamma)
    d.compute(data, FLAGS.eps)
    ratio = d.get_W()
    dice = estimators.DiscreteDice(ratio)

    print ('{} est ratios'.format(method))
    if 'abs' in method:
        est = dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'])
    else:
        est = dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'])
    return est

def off_estimate_dual(seed, data, gamma, mdp, pie, method):
    d = TabularDualDice(method, gamma, mdp, pie)
    d.compute(data, FLAGS.eps)
    ratio = d.get_W()
    dice = estimators.DiscreteDice(ratio)

    print ('{} est ratios'.format(method))
    if 'abs' in method:
        est = dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'])
    else:
        est = dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'])

    return est, ratio

def off_dual_dens_estimate(seed, data, true_dens, gamma, mdp, pie, method):

    _, ratios = off_estimate_dual(seed, data, gamma, mdp, pie, method)
    if 'abs' in method:
        est_dens = true_dens['abs_pib'] * ratios
    else:
        est_dens = true_dens['g_pib'] * ratios
    return est_dens

def off_estimate_learn(seed, data, gamma, mdp, method, alpha_w = False, alpha_r = False, unit_mean = True, uh_stabilizer = False):
    # abstraction case
    abs_d_ratio = TabularDensityRatioEstimationGAN(seed, mdp.n_state, mdp.n_action,
                                                gamma,  
                                                q_hidden_layers = -1, q_hidden_dim = -1,\
                                                w_hidden_layers = -1, w_hidden_dim = -1,\
                                                activation = None, 
                                                abs_state_dims = mdp.n_abs_s,
                                                alpha_w = alpha_w, alpha_r = alpha_r,
                                                unit_mean = unit_mean,
                                                method = method, Q_lr = FLAGS.Q_lr,
                                                W_lr = FLAGS.W_lr, lam_lr = FLAGS.lam_lr, uh_stabilizer = uh_stabilizer)
    abs_d_ratio.train(data, epochs = FLAGS.epochs, print_log = True) 
    abstract_ratio = abs_d_ratio.get_W()
    abs_dice = estimators.Dice(abstract_ratio)
    if method == 'ground':
        #abs_est = abs_dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'], temp = data['abs_state_b_act_b'])
        abs_est = abs_dice.estimate(data['state_b'], data['rewards'], data['gammas'])
    else:
        #abs_est = abs_dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'], temp = data['state_b_act_b'])
        abs_est = abs_dice.estimate(data['abs_state_b'], data['rewards'], data['gammas'])
    return abs_est

def off_estimate_sarsa(seed, data, gamma, mdp, pie, method, abs_pie = None, true_dens = None):
    fqe = DiscreteFQE(method, mdp, gamma, pie, abs_pie)
    q = fqe.compute(data) 
    
    val = 0
    if 'abs' in method:
        for a in range(mdp.n_action):
            prob = 1. * abs_pie.get_prob(0, a)
            idx = mdp.get_sa_encoding(0, a)
            val += prob * q[idx]
    else:
        init_states = data['init_states']
        for s in init_states:
            for a in range(mdp.n_action):
                prob = init_states[s] * pie.get_prob(s, a)
                if 'abs' in method:
                    idx = mdp.get_sa_encoding(mdp.phi(s), a)
                else:
                    idx = mdp.get_sa_encoding(s, a)
                val += prob * q[idx]
    temp = np.sum([gamma ** i for i in range(FLAGS.traj_len)])
    return (1. - gamma) * val

def main():  # noqa
    batch_size = FLAGS.batch_size
    traj_len = FLAGS.traj_len
    
    mdp, gamma = env_setup(), FLAGS.gamma
    pie, pib = policies_setup(mdp)

    mdp.set_pie(pie)
    mdp.set_gamma(gamma)

    seed = FLAGS.seed
    np.random.seed(seed)

    #data, _, _, _ = utils.collect_data_discrete(mdp, pie, traj_len, 200, gamma = gamma)
    #utils.compute_Q_func(mdp, data, gamma)

    np.random.seed(seed)
    oracle_est = on_policy_estimate(seed, FLAGS.oracle_batch_size, traj_len, gamma, mdp, pie)
    np.random.seed(seed)
    on_policy_estimate(seed, FLAGS.oracle_batch_size, traj_len, gamma, mdp, pib)

    np.random.seed(seed)
    _, _, _, g_pie_densities = utils.collect_data_discrete(mdp, pie, traj_len, 300, gamma = gamma)
    
    abs_pie = None
    #abs_pie = AbstractDiscreteGridworld(mdp, pie, g_pie_densities)
    #np.random.seed(seed)
    #on_policy_estimate(seed, FLAGS.oracle_batch_size, traj_len, gamma, mdp, abs_pie)

    true_dens = {} 
    if FLAGS.compute_true_ratios:
        _, _, _, g_pib_densities = utils.collect_data_discrete(mdp, pib, traj_len, 300, gamma = None)

        g_pie_pi_densities = utils.compute_dsa(mdp, g_pie_densities, pie) 
        g_pib_pi_densities = utils.compute_dsa(mdp, g_pib_densities, pib) 

        #abs_pib_model = utils.compute_abs_model(mdp, g_pib_densities)
        #abs_pie_model = utils.compute_abs_model(mdp, g_pie_densities)

        true_dens['g_pie'] = g_pie_pi_densities
        true_dens['g_pib'] = g_pib_pi_densities
        true_dens['abs_pie'] = utils.compute_abs_densities(g_pie_densities, pie, mdp)
        true_dens['abs_pib'] = utils.compute_abs_densities(g_pib_densities, pib, mdp)

    if FLAGS.exp_name == 'ope':
        algos = ['True Ground', 'True Abs', 'MWLGround', 'MWLAbstract', 'MWLAbstractPract']
        r_ests = run_experiment(seed, batch_size, traj_len, gamma, mdp, pie, pib, abs_pie, true_dens = true_dens)

        assert len(algos) == len(r_ests)

        mses = []
        for r in r_ests:
            mses.append(utils.get_MSE([oracle_est], [r]))

        summary = {
            'results': {},
            'seed': seed,
            'batch_size': batch_size,
            'traj_len': traj_len,
            'hp': {
                'eps': FLAGS.eps,
                'Q_lr': FLAGS.Q_lr,
                'W_lr': FLAGS.W_lr,
                'lam_lr': FLAGS.lam_lr
            },
            'oracle_est': oracle_est
        }

        for idx, algo in enumerate(algos):
            summary['results'][algo] = {
                'mse': mses[idx]['mean'], # single MSE just for single trial
                'r_est': r_ests[idx]
            }
    elif FLAGS.exp_name == 'densities':
        algos = ['TrueDensities', 'EstDensities']
        dens_ests = run_experiment_densities(seed, batch_size, traj_len, gamma, mdp, pie, pib, abs_pie, true_dens = true_dens)
        summary = {
            'results': {},
            'seed': seed,
            'batch_size': batch_size,
            'traj_len': traj_len,
            'hp': {
                'eps': FLAGS.eps,
                'Q_lr': FLAGS.Q_lr,
                'W_lr': FLAGS.W_lr,
                'lam_lr': FLAGS.lam_lr
            },
            'oracle_est': oracle_est
        }
        for idx, algo in enumerate(algos):
            summary['results'][algo] = {
                'dens_est': dens_ests[idx]
            }

    print (summary)
    np.save(FLAGS.outfile, summary) 

if __name__ == '__main__':
    main()
