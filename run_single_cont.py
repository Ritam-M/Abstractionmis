from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pdb
import argparse
import random

from infinite_walker import InfiniteWalker2d
from infinite_antumaze import InfiniteAntUMaze
from infinite_pusher import InfinitePusher
from infinite_reacher import InfiniteReacher
from a2c_ppo_acktr.model import load_policy
from policies import GMixPolicy, DeepOPEPolicy
import estimators
import utils
from continuous_density_ratio import ContinuousDensityRatioEstimationGAN, ContinuousDensityRatioEstimationKernel

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
parser.add_argument('--mdp_num', default = 0, type = int)
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--pi_set', default = 0, type = int)
parser.add_argument('--mix_ratio', default = 0.7, type = float)
parser.add_argument('--epochs', default = 2000, type = int)
parser.add_argument('--oracle_batch_size', default = 5, type = int)
parser.add_argument('--pib_est', default = 'true', type = str2bool)

# variables
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--batch_size', default = 5, type = int)
parser.add_argument('--traj_len', default = 5, type = int)
parser.add_argument('--Q_lr', default = 1e-4, type = float)
parser.add_argument('--W_lr', default = 1e-4, type = float)
parser.add_argument('--lam_lr', default = 1e-4, type = float)

# misc
parser.add_argument('--plot', default = 'false', type = str2bool)
parser.add_argument('--exp_name', default = 'gan', type = str)
parser.add_argument('--print_log', default = 'false', type = str2bool)

FLAGS = parser.parse_args()

def env_setup():
    if FLAGS.env_name == 'Reacher':
        if FLAGS.mdp_num == 0:
            env = InfiniteReacher()
    elif FLAGS.env_name == 'Pusher':
        if FLAGS.mdp_num == 0:
            env = InfinitePusher()
    elif FLAGS.env_name == 'AntUMaze':
        if FLAGS.mdp_num == 0:
            from d4rl.locomotion import wrappers
            env = wrappers.NormalizedBoxEnv(InfiniteAntUMaze())
    elif FLAGS.env_name == 'Walker':
        if FLAGS.mdp_num == 0:
            env = InfiniteWalker2d()
    return env

def policies_setup(env):
    mix_ratios = [FLAGS.mix_ratio, 1. - FLAGS.mix_ratio]
    if FLAGS.env_name == 'Reacher':
        if FLAGS.pi_set == 0:
            pie = load_policy('reacher/policies/150', std = np.array([0.1, 0.1]))
            pi = load_policy('reacher/policies/150', std = np.array([0.5, 0.5]))
            pis = [pi, pie]
            pib = GMixPolicy(pis, mix_ratios)
    elif FLAGS.env_name == 'Pusher':
        if FLAGS.pi_set == 0:
            pie = load_policy('pusher/policies/300', std = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
            pi = load_policy('pusher/policies/300', std = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
            pis = [pi, pie]
            pib = GMixPolicy(pis, mix_ratios)
    elif FLAGS.env_name == 'AntUMaze':
        if FLAGS.pi_set == 0:
            pie = DeepOPEPolicy('antumaze/policies/10.pkl', env.action_space.shape[0], std = 0.1)
            pib = DeepOPEPolicy('antumaze/policies/5.pkl', env.action_space.shape[0], std = 0.1)
    elif FLAGS.env_name == 'Walker':
        if FLAGS.pi_set == 0:
            pie_std = np.array([0.1 for _ in range(6)])
            pie = load_policy('walker/policies/460', std = pie_std)
            pib_std = np.array([0.5 for _ in range(6)])
            pib = load_policy('walker/policies/460', std = pib_std)
    return pie, pib

def _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib):
    np.random.seed(seed)
    initial_states = mdp.get_initial_state_samples(max(batch_size * truncated_horizon, 10000))

    np.random.seed(seed)
    torch.manual_seed(seed)
    g_paths, _ = utils.collect_data(mdp, pib, batch_size, truncated_horizon = truncated_horizon)

    data = {
        'initial_states': initial_states,
        'ground_data': g_paths
    }
    # format data into relevant inputs needed by loss function
    np.random.seed(seed)
    data = utils.format_data_new(data, mdp.phi, gamma)
    return data

# Q as neural net
def run_experiment_gan_rank(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib)

    # best dice
    np.random.seed(seed)
    g_bdice_est, g_bdice_metrics = off_ground_estimate(seed, data, gamma, mdp, pie, pib, alpha_w = True, alpha_r = True, rank_penalty = False)
   
    np.random.seed(seed)
    g_bdice_rank_est, g_bdice_rank_metrics = off_ground_estimate(seed, data, gamma, mdp, pie, pib, alpha_w = True, alpha_r = True, rank_penalty = True)

    results = [g_bdice_est, g_bdice_rank_est]
    metrics = [g_bdice_metrics, g_bdice_rank_metrics]
    print (results)
    return results, metrics

# Q as neural net
def run_experiment_gan(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib)

    # best dice version of dual dice
    np.random.seed(seed)
    g_ddice_est, g_ddice_metrics = off_ground_estimate(seed, data, gamma, mdp, pie, pib, alpha_w = True, unit_norm = True)
   
    np.random.seed(seed)
    abs_ddice_est, abs_ddice_metrics = off_abstract_estimate(seed, data, gamma, mdp, pie, pib, alpha_w = True, unit_norm = True) 

    results = [g_ddice_est, abs_ddice_est]
    metrics = [g_ddice_metrics, abs_ddice_metrics]
    print (results)
    return results, metrics

# RHKS experiments
def run_experiment_rkhs(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib):

    data = _data_prep(seed, batch_size, truncated_horizon, gamma, mdp, pie, pib)

    # mwl with RBF kernel
    np.random.seed(seed)
    g_mwl_est = off_ground_estimate(seed, data, gamma, mdp, pie, pib, gan = False)
   
    np.random.seed(seed)
    abs_mwl_est = off_abstract_estimate(seed, data, gamma, mdp, pie, pib, gan = False)

    results = (g_mwl_est, abs_mwl_est)
    print (results)
    return results

def on_policy_estimate(seed, batch_size, truncated_horizon, gamma, mdp, pie):
    # on-policy ground case
    g_pie_paths, _ = utils.collect_data(mdp, pie, batch_size, truncated_horizon)
    g_pie_estimator = estimators.OnPolicy(pie, gamma)
    g_pie_est = g_pie_estimator.estimate(g_pie_paths)
    return g_pie_est

# ground MWL and BestDice
def off_ground_estimate(seed, data, gamma, mdp, pie, pib, gan = True, alpha_w = False,
                        alpha_r = False, rank_penalty = False, unit_norm = False,
                        uh_stabilizer = False, w_pos = True):
    # ground case
    if gan:
        g_d_ratio = ContinuousDensityRatioEstimationGAN(seed, mdp.state_dims,
                                                    mdp.action_dims,\
                                                    gamma, pie = pie,
                                                    w_hidden_layers = 2, w_hidden_dim = 64,
                                                    q_hidden_layers = 2, q_hidden_dim = 64,\
                                                    activation = 'tanh', alpha_w = alpha_w,\
                                                    alpha_r = alpha_r, method = 'ground', Q_lr = FLAGS.Q_lr,\
                                                    W_lr = FLAGS.W_lr, lam_lr = FLAGS.lam_lr,
                                                    rank_penalty = rank_penalty, uh_stabilizer = uh_stabilizer,
                                                    unit_norm = unit_norm, w_pos = w_pos)

    else:
        g_d_ratio = ContinuousDensityRatioEstimationKernel(seed, mdp.state_dims,
                                                        mdp.action_dims,\
                                                        gamma, pie = pie, hidden_layers = 2,\
                                                        hidden_dim = 32, activation = 'relu',\
                                                        method = 'ground', W_lr = FLAGS.W_lr)
    g_d_ratio.train(data, epochs = FLAGS.epochs, print_log = FLAGS.print_log)
    ground_ratio = g_d_ratio.get_W()
    g_dice = estimators.Dice(ground_ratio)
    g_est = g_dice.estimate(data['state_b_act_b'], data['rewards'], data['gammas'])
    return g_est, g_d_ratio.get_metrics()

# abstract MWL and BestDice
def off_abstract_estimate(seed, data, gamma, mdp, pie, pib, gan = True, alpha_w = False,
                        alpha_r = False, rank_penalty = False, uh_stabilizer = False,
                        unit_norm = False, w_pos = True):
    # abstraction case
    if gan:
        abs_d_ratio = ContinuousDensityRatioEstimationGAN(seed, mdp.state_dims, mdp.action_dims,
                                                    gamma, pie = pie, 
                                                    w_hidden_layers = 2, w_hidden_dim = 64,\
                                                    q_hidden_layers = 2, q_hidden_dim = 64,\
                                                    activation = 'tanh', 
                                                    abs_state_dims = mdp.abs_state_dims,
                                                    alpha_w = alpha_w, alpha_r = alpha_r,
                                                    method = 'abs_new', Q_lr = FLAGS.Q_lr,
                                                    W_lr = FLAGS.W_lr, lam_lr = FLAGS.lam_lr, rank_penalty = rank_penalty,
                                                    uh_stabilizer = uh_stabilizer, unit_norm = unit_norm, w_pos = w_pos)
    else:
        abs_d_ratio = ContinuousDensityRatioEstimationKernel(seed, mdp.state_dims, mdp.action_dims,
                                                    gamma, pie = pie, hidden_layers = 2,
                                                    hidden_dim = 32, activation = 'relu', 
                                                    abs_state_dims = mdp.abs_state_dims,
                                                    method = 'abs_ori', W_lr = FLAGS.W_lr)

    abs_d_ratio.train(data, epochs = FLAGS.epochs, print_log = FLAGS.print_log) 
    abstract_ratio = abs_d_ratio.get_W()
    abs_dice = estimators.Dice(abstract_ratio)
    abs_est = abs_dice.estimate(data['abs_state_b_act_b'], data['rewards'], data['gammas'])
    return abs_est, abs_d_ratio.get_metrics()

def main():  # noqa
    batch_size = FLAGS.batch_size
    traj_len = FLAGS.traj_len
    
    mdp, gamma = env_setup(), FLAGS.gamma
    pie, pib = policies_setup(mdp)

    seed = FLAGS.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    oracle_est = on_policy_estimate(seed, FLAGS.oracle_batch_size, traj_len, gamma, mdp, pie)
    print ('pie true: {}'.format(oracle_est))
    if FLAGS.pib_est:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pib_est = on_policy_estimate(seed, FLAGS.oracle_batch_size, traj_len, gamma, mdp, pib)
        print ('pib true: {}'.format(pib_est))

    if FLAGS.exp_name == 'gan':
        algos = ['BestDICE', 'AbstractBestDICE']
        r_ests, metrics = run_experiment_gan(seed, batch_size, traj_len, gamma, mdp, pie, pib)
        #r_ests, metrics = run_experiment_gan_rank(seed, batch_size, traj_len, gamma, mdp, pie, pib)
        r_ests_training = [m['r_ests'] for m in metrics]
        q_ranks_training = [m['q_ranks'] for m in metrics]
        w_ranks_training = [m['w_ranks'] for m in metrics]
    elif FLAGS.exp_name == 'rkhs':
        algos = ['MWL (rkhs)', 'Abs MWL (rkhs)']
        r_ests = run_experiment_rkhs(seed, batch_size, traj_len, gamma, mdp, pie, pib)

    rel_den = utils.get_MSE([pib_est], [oracle_est])['mean']
    mses = []
    for r in r_ests:
        mse = utils.get_MSE([oracle_est], [r])['mean']
        mses.append(mse / rel_den)
    mses_training = []
    for r in r_ests_training:
        mses_training.append(np.square(oracle_est - r) / rel_den) # average across trials in plotting
    
    summary = {
        'results': {},
        'seed': seed,
        'batch_size': batch_size,
        'traj_len': traj_len,
        'hp': {
            'Q_lr': FLAGS.Q_lr,
            'W_lr': FLAGS.W_lr,
            'lam_lr': FLAGS.lam_lr
        },
        'oracle_est': oracle_est
    }

    for idx, algo in enumerate(algos):
        summary['results'][algo] = {
            'mse': mses[idx], # single MSE just for single trial
            'r_est': r_ests[idx],
            'mse_training': mses_training[idx],
            'r_ests': r_ests_training[idx],
            'q_ranks': q_ranks_training[idx],
            'q_rank': q_ranks_training[idx][-1],
            'w_ranks': w_ranks_training[idx],
            'w_rank': w_ranks_training[idx][-1]
        }
    print (summary)
    np.save(FLAGS.outfile, summary) 

if __name__ == '__main__':
    main()
