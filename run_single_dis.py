from __future__ import print_function
from __future__ import division

import numpy as np
np.set_printoptions(suppress=True)
import pdb
import argparse
import random

from toymdp import GraphMDP
from policies import ToyMDPPolicy
import estimators
import utils
from discrete_density_ratio import DiscreteDensityRatioEstimation, TabularDualDice, TabularStateCOPTD

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
        elif FLAGS.mdp_num == 4:
            phi = {
                0: 0,
                1: 1,
                2: 2,
                3: 2,
                4: 3,
            }
            abstraction = {
                'phi': phi,
                'num_abs_states': 4
            }
            num_states = 5
            num_actions = 2
            rewards = np.zeros((num_states, num_actions))
            rewards[2, 0] = 1
            rewards[3, 0] = 1
            env = GraphMDP(num_states, num_actions, rewards = rewards, init_state = 0, abstraction = abstraction)
            
            # setup MDP
            env.set_transition_prob(0, 0, 1, 1.0)
            env.set_transition_prob(1, 0, 2, 1.0)
            env.set_transition_prob(1, 1, 3, 1.0)
            env.set_transition_prob(2, 0, 4, 1.0)
            env.set_transition_prob(2, 1, 3, 1.0)
            env.set_transition_prob(3, 0, 2, 1.0)
            env.set_transition_prob(3, 1, 4, 1.0)
            env.set_transition_prob(4, 0, 1, 1.0)
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
        elif FLAGS.pi_set == 2:
            num_states = env.n_state
            num_actions = env.n_action
            pie = np.zeros((num_states, num_actions))
            pie[0][0] = 1.
            pie[1][0] = 0.01
            pie[1][1] = 0.99
            pie[2][0] = 0.9
            pie[2][1] = 0.1
            pie[3][0] = 0.1
            pie[3][1] = 0.9
            pie[4][0] = 1.

            pie = ToyMDPPolicy(num_actions, pie)

            pib = np.zeros((num_states, num_actions))
            pib[0][0] = 1.
            pib[1][0] = 0.99
            pib[1][1] = 0.01
            pib[2][0] = 0.1
            pib[2][1] = 0.9
            pib[3][0] = 0.9
            pib[3][1] = 0.1
            pib[4][0] = 1.
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
    data = utils.format_data_discrete(mdp, data, pie, gamma, abs_pie, pib = pib)
    return data

def run_experiment_coptd(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie, true_dens = None):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie)

    # true densities
    np.random.seed(seed)
    true_g_est, true_a_est = true_off_estimate(seed, data, gamma, true_dens)

    np.random.seed(seed)
    g_est, _ = off_estimate_coptd(seed, data, gamma, mdp, method = 'ground')
    
    np.random.seed(seed)
    abs_est_pract, _ = off_estimate_coptd(seed, data, gamma, mdp, method = 'abs-pract')
    
    results = (true_g_est, true_a_est, g_est, abs_est_pract)
    print (results)
    return results

def run_experiment(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie, true_dens = None):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib, abs_pie)

    # true densities
    np.random.seed(seed)
    true_g_est, true_a_est = true_off_estimate(seed, data, gamma, true_dens)

    np.random.seed(seed)
    g_est_mwl, _ = off_estimate_dual(seed, data, gamma, mdp, pie, method = 'ground')
    
    np.random.seed(seed)
    abs_est_mwl_pract, _ = off_estimate_dual(seed, data, gamma, mdp, pie, method = 'abs-pract')
    
    results = (true_g_est, true_a_est, g_est_mwl, abs_est_mwl_pract)
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

    # state
    d_pie = true_dens['state_g_pie'][np.unique(np.argmax(data['state_b'], axis = 1))]
    d_pib = true_dens['state_g_pib'][np.unique(np.argmax(data['state_b'], axis = 1))]
    x = d_pie / d_pib
    print ('gr pie state densities {}'.format(d_pie))
    print ('gr pib state densities {}'.format(d_pib))
    print ('true ground state ratios {}'.format(x))
    print ('weighted true ground ratios {}'.format(x / np.sum(x)))

    # state-action
    ge_t = true_dens['g_pie'][np.argmax(data['state_b_act_b'], axis = 1)]
    gb_t = true_dens['g_pib'][np.argmax(data['state_b_act_b'], axis = 1)]
    gr = ge_t / gb_t
    d_pie = true_dens['g_pie'][np.unique(np.argmax(data['state_b_act_b'], axis = 1))]
    d_pib = true_dens['g_pib'][np.unique(np.argmax(data['state_b_act_b'], axis = 1))]
    
    x = d_pie / d_pib
    #print ('gr pie densities {}'.format(d_pie))
    #print ('gr pib densities {}'.format(d_pib))
    #print ('true ground ratios {}'.format(x))
    g_est = np.mean(gr * rews)
    #print ('est with true gr {}'.format(g_est))

    # state
    d_abs_pie = true_dens['state_abs_pie'][np.unique(np.argmax(data['abs_state_b'], axis = 1))]
    d_abs_pib = true_dens['state_abs_pib'][np.unique(np.argmax(data['abs_state_b'], axis = 1))]
    y = d_abs_pie / d_abs_pib
    print ('abs pie state densities {}'.format(d_abs_pie))
    print ('abs pib state densities {}'.format(d_abs_pib))
    print ('true abs state ratios {}'.format(y))
    print ('weighted true abs state ratios {}'.format(y / np.sum(y)))

    # state-action
    ae_t = true_dens['abs_pie'][np.argmax(data['abs_state_b_act_b'], axis = 1)]
    ab_t = true_dens['abs_pib'][np.argmax(data['abs_state_b_act_b'], axis = 1)]
    ar = ae_t / ab_t
    d_abs_pie = true_dens['abs_pie'][np.unique(np.argmax(data['abs_state_b_act_b'], axis = 1))]
    d_abs_pib = true_dens['abs_pib'][np.unique(np.argmax(data['abs_state_b_act_b'], axis = 1))]
    y = d_abs_pie / d_abs_pib
    #print ('abs pie densities {}'.format(d_abs_pie))
    #print ('abs pib densities {}'.format(d_abs_pib))
    #print ('true abs ratios {}'.format(y))
    a_est = np.mean(ar * rews)
    #print ('est with true abs {}'.format(a_est))

    return g_est, a_est

def off_estimate_coptd(seed, data, gamma, mdp, method):
    d = TabularStateCOPTD(method, gamma, mdp)
    d.compute(data)
    ratio = d.get_W()
    pdb.set_trace()
    return 0,0

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

def main():  # noqa
    batch_size = FLAGS.batch_size
    traj_len = FLAGS.traj_len
    
    mdp, gamma = env_setup(), FLAGS.gamma
    pie, pib = policies_setup(mdp)

    mdp.set_pie(pie)
    mdp.set_gamma(gamma)

    seed = FLAGS.seed
    np.random.seed(seed)

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

        true_dens['state_g_pie'] = g_pie_densities
        true_dens['state_g_pib'] = g_pib_densities
        true_dens['state_abs_pie'] = utils.compute_abs_densities(g_pie_densities, pie, mdp, state_only = True)
        true_dens['state_abs_pib'] = utils.compute_abs_densities(g_pib_densities, pib, mdp, state_only = True)
        true_dens['g_pie'] = g_pie_pi_densities
        true_dens['g_pib'] = g_pib_pi_densities
        true_dens['abs_pie'] = utils.compute_abs_densities(g_pie_densities, pie, mdp)
        true_dens['abs_pib'] = utils.compute_abs_densities(g_pib_densities, pib, mdp)

    if FLAGS.exp_name == 'ope_dice':
        algos = ['True Ground', 'True Abs', 'MWLGround', 'MWLAbstract']
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
    elif FLAGS.exp_name == 'ope_coptd':
        algos = ['True Ground', 'True Abs', 'COPTDGround', 'COPTDAbstract']
        r_ests = run_experiment_coptd(seed, batch_size, traj_len, gamma, mdp, pie, pib, abs_pie, true_dens = true_dens)

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
