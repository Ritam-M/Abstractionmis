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
parser.add_argument('--oracle_batch_size', default = 500, type = int)

# variables
parser.add_argument('--num_trials', default = 1, type=int, help='The number of trials to launch.')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')
parser.add_argument('--exp_name', default = 'tab', type = str)

FLAGS = parser.parse_args()

ct = 0
EXECUTABLE = 'exp_dis.sh'

def get_cmd(exp_name,
            seed,
            outfile,
            batch_size,
            traj_len,
            qlr, wlr, lamlr,
            condor = False):
  
    arguments = '--outfile %s --seed %d' % (outfile, seed)
   
    arguments += ' --exp_name %s' % exp_name
    arguments += ' --env_name %s' % FLAGS.env_name
    arguments += ' --mdp_num %d' % FLAGS.mdp_num
    arguments += ' --gamma %f' % FLAGS.gamma
    arguments += ' --pi_set %d' % FLAGS.pi_set
    arguments += ' --mix_ratio %f' % FLAGS.mix_ratio
    arguments += ' --oracle_batch_size %d' % FLAGS.oracle_batch_size
    
    arguments += ' --epochs %d' % FLAGS.epochs
    arguments += ' --Q_lr %f' % qlr 
    arguments += ' --W_lr %f' % wlr 
    arguments += ' --lam_lr %f' % lamlr

    arguments += ' --batch_size %d' % batch_size
    arguments += ' --traj_len %d' % traj_len

    if FLAGS.condor:
        cmd = '%s' % (arguments)
    else:
        EXECUTABLE = 'run_single_dis.py'
        cmd = 'python3 %s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(exp_name, seed,
            outfile,
            batch_size,
            traj_len,
            qlr, wlr, lamlr,
            condor = False):

    cmd = get_cmd(exp_name, seed,
                outfile,
                batch_size,
                traj_len, qlr, wlr, lamlr)
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

        setup_files = 'http://proxy.chtc.wisc.edu/SQUID/pavse/research.tar.gz'
        common_main_files = 'run_single_dis.py, discrete_density_ratio.py, estimators.py, policies.py, utils.py'
        domains = 'toymdp.py'

        submitFile += 'transfer_input_files = {}, {}, {}\n'.format(setup_files, common_main_files, domains)
        submitFile += 'requirements = (has_avx == True)\n'
        submitFile += 'request_cpus = 1\n'
        submitFile += 'request_memory = 5GB\n'
        submitFile += 'request_disk = 5GB\n'
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

def _launch_trial(exp_name, seeds, b, t, q, w, lam):

    global ct
    for seed in seeds: 
        #outfile = 'env_{}_exp_{}_seed_{}_batch_{}_traj_{}_mix_{}.npy'.format(FLAGS.env_name, exp_name, seed, b, t, FLAGS.mix_ratio)
        outfile = 'env_{}_exp_{}_seed_{}_batch_{}_traj_{}_mix_{}_Qlr_{}_Wlr_{}_lamlr_{}.npy'.format(FLAGS.env_name + str(FLAGS.mdp_num), exp_name, seed, b, t, FLAGS.mix_ratio, q, w, lam)
        if os.path.exists(outfile):
            continue
        run_trial(exp_name, seed,
                outfile,
                batch_size = b,
                traj_len = t,
                qlr = q,
                wlr = w,
                lamlr = lam,
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

    #batch_sizes = [1, 5, 10, 50, 100, 200, 300, 500]
    batch_sizes = [300]
    traj_lens = [100]

    Q_lrs = [0] 
    W_lrs = [0] 
    lam_lrs = [0]

    data_combined = [batch_sizes, traj_lens, Q_lrs, W_lrs, lam_lrs]
    data_combined = list(itertools.product(*data_combined))

    for e in data_combined:
        b, t, q, w, lam = e
        _launch_trial(FLAGS.exp_name, seeds, b, t, q, w, lam)

    print('%d experiments ran.' % ct)

if __name__ == "__main__":
    main()


