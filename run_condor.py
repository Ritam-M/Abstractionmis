from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time
import pdb
import itertools

parser = argparse.ArgumentParser()
# saving
parser.add_argument('result_directory', default = None, help='Directory to write results to.')

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--mdp_num', default = 0, type = int, required = True)
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--pi_set', default = 0, type = int, required = True)
parser.add_argument('--mix_ratio', default = 0.7, type = float)
parser.add_argument('--epochs', default = 2000, type = int)
parser.add_argument('--oracle_batch_size', default = 200, type = int)

# variables
parser.add_argument('--num_trials', default = 1, type=int, help='The number of trials to launch.')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')
parser.add_argument('--exp_name', default = 'gan', type = str)

FLAGS = parser.parse_args()

ct = 0
EXECUTABLE = 'exp.sh'

def get_cmd(exp_name,
            seed,
            outfile,
            batch_size,
            traj_len,
            Q_lr, W_lr, lam_lr,
            condor = False):
  
    arguments = '--outfile %s --seed %d' % (outfile, seed)
   
    arguments += ' --exp_name %s' % exp_name
    arguments += ' --env_name %s' % FLAGS.env_name
    arguments += ' --mdp_num %d' % FLAGS.mdp_num
    arguments += ' --gamma %f' % FLAGS.gamma
    arguments += ' --pi_set %d' % FLAGS.pi_set
    arguments += ' --mix_ratio %f' % FLAGS.mix_ratio
    arguments += ' --epochs %d' % FLAGS.epochs
    arguments += ' --oracle_batch_size %d' % FLAGS.oracle_batch_size

    arguments += ' --batch_size %d' % batch_size
    arguments += ' --traj_len %d' % traj_len

    arguments += ' --W_lr %f' % W_lr
    arguments += ' --Q_lr %f' % Q_lr
    arguments += ' --lam_lr %f' % lam_lr
   
    if FLAGS.condor:
        cmd = '%s' % (arguments)
    else:
        EXECUTABLE = 'run_single_cont.py'
        cmd = 'python3 %s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(exp_name, seed,
            outfile,
            batch_size,
            traj_len,
            Q_lr, W_lr, lam_lr,
            condor = False):

    cmd = get_cmd(exp_name, seed,
                outfile,
                batch_size,
                traj_len,
                Q_lr, W_lr, lam_lr)
    if condor:
        submitFile = 'universe = vanilla\n'
        submitFile += 'executable = ' + EXECUTABLE + "\n"
        submitFile += 'arguments = ' + cmd + '\n'
        submitFile += 'error = %s.err\n' % outfile
        #submitFile += 'log = %s.log\n' % outfile
        submitFile += 'log = /dev/null\n'
        submitFile += 'output = /dev/null\n'
        #submitFile += 'output = %s.out\n' % outfile
        submitFile += 'should_transfer_files = YES\n'
        submitFile += 'when_to_transfer_output = ON_EXIT\n'

        setup_files = 'rpm, mujoco_setup.sh, http://proxy.chtc.wisc.edu/SQUID/pavse/research.tar.gz'
        common_main_files = 'run_single_cont.py, continuous_density_ratio.py, estimators.py, policies.py, utils.py'
        domains = 'infinite_walker.py, walker, infinite_pusher.py, pusher, infinite_antumaze.py, antumaze, infinite_reacher.py, reacher, a2c_ppo_acktr'

        submitFile += 'transfer_input_files = {}, {}, {}\n'.format(setup_files, common_main_files, domains)
        submitFile += 'requirements = (has_avx == True)\n'
        submitFile += 'request_cpus = 1\n'
        submitFile += 'request_memory = 5GB\n'
        submitFile += 'request_disk = 7GB\n'
        submitFile += 'queue'

        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.2)
    else:
        # TODO
        pdb.set_trace()
        #subprocess.run('"conda init bash; conda activate research; {}"'.format(cmd), shell=True)
        #cmd = 'bash -c "source activate root"' 
        subprocess.Popen(('conda run -n research ' + cmd).split())

def _launch_trial(exp_name, combined_hp, seeds, b, t):

    global ct
    for e in combined_hp:
        #qlr, wlr, lamlr = e
        lr, lamlr = e
        qlr = lr
        wlr = lr
        for seed in seeds: 
            outfile = 'env_{}_exp_{}_seed_{}_batch_{}_traj_{}_mix_{}_Qlr_{}_Wlr_{}_lamlr_{}.npy'.format(FLAGS.env_name, exp_name, seed, b, t, FLAGS.mix_ratio, qlr, wlr, lamlr)
            if os.path.exists(outfile):
                continue
            run_trial(exp_name, seed,
                    outfile,
                    batch_size = b,
                    traj_len = t,
                    Q_lr = qlr,
                    W_lr = wlr,
                    lam_lr = lamlr,
                    condor = FLAGS.condor)
            ct += 1
            print ('submitted job number: %d' % ct)

def main():  # noqa
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory + '_' + FLAGS.env_name + '_' + FLAGS.exp_name + '_mdp_' + str(FLAGS.mdp_num) + '_pi_' + str(FLAGS.pi_set) + '_mix_' + str(FLAGS.mix_ratio)

    if not os.path.exists(directory):
        os.makedirs(directory)
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    batch_sizes = [5, 10, 50, 75, 100, 300, 500, 1000]
    traj_lens = [500]

    data_combined = [batch_sizes, traj_lens]
    data_combined = list(itertools.product(*data_combined))

    gan_lrs = [5e-5, 1e-4, 3e-4, 7e-4, 1e-3]
    gan_lam_lrs = [1e-3]
    gan_combined = [
        gan_lrs,
        gan_lam_lrs
    ]
    gan_combined = list(itertools.product(*gan_combined))

    rkhs_W_lrs = [5e-6, 1e-5, 5e-5]
    rkhs_Q_lrs = [0]
    rkhs_lam_lrs =[0]
    rkhs_combined = [
        rkhs_Q_lrs,
        rkhs_W_lrs,
        rkhs_lam_lrs
    ]
    rkhs_combined = list(itertools.product(*rkhs_combined))
  
    for e in data_combined:
        b, t = e
        if FLAGS.exp_name == 'gan':
            _launch_trial('gan', gan_combined, seeds, b, t)
        elif FLAGS.exp_name == 'rkhs':
            _launch_trial('rkhs', rkhs_combined, seeds, b, t)
        elif FLAGS.exp_name == 'gan-rkhs': 
            _launch_trial('gan', gan_combined, seeds, b, t)
            _launch_trial('rkhs', rkhs_combined, seeds, b, t)

    print('%d experiments ran.' % ct)

if __name__ == "__main__":
    main()


