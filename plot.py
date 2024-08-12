"""Plot value function results."""
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import pdb
from matplotlib import rcParams
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bool_argument(parser, name, default=False, msg=''):
    dest = name.replace('-', '_')
    parser.add_argument('--%s' % name, dest=dest, type=bool, default=default, help=msg)
    parser.add_argument('--no-%s' % name, dest=dest, type=bool, default=default, help=msg)

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', help=help)
parser.add_argument('--domain', type = str)
parser.add_argument('--plot_type', type = str)
parser.add_argument('--vs', type = str)
parser.add_argument('--other_fixed_val', type = int)
parser.add_argument('--training_graph', type = str2bool)
parser.add_argument('--tr_metric', type = str)
parser.add_argument('--batch', type = int)
parser.add_argument('--traj', type = int)
parser.add_argument('--vs_hp', type = str)
parser.add_argument('--qlr', type = float)
parser.add_argument('--wlr', type = float)
parser.add_argument('--lamlr', type = float)
parser.add_argument('--method', type = str)
FLAGS = parser.parse_args()

def read_proto(filename):
    results = results_pb2.MethodResult()
    with open(filename, 'rb') as f:
        results.ParseFromString(f.read())
    return results

def plot_vs(data, vs, other_fixed_val, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 12.0, forward=True)

    for method in sorted(data):
        label = method
        print ('method: {}'.format(method))
        if vs == 'traj':
            it = data[method][other_fixed_val]
        elif vs == 'batch':
            it = data[method]

        x = []
        y = []
        ylower = []
        yupper = []
        for vs_val in it:
            if vs == 'traj':
                errors = np.array(it[vs_val]['mse'])
                mean_rew = np.mean(it[vs_val]['r_est'])
            elif vs == 'batch':
                errors = np.array(it[vs_val][other_fixed_val]['mse'])
                mean_rew = np.mean(it[vs_val][other_fixed_val]['r_est'])
            n = len(errors) # number of trials
            mean = np.mean(errors)
            std = np.std(errors)

            yerr = 1.96 * std / np.sqrt(float(n))
            y.append(mean)
            ylower.append(mean - yerr)
            yupper.append(mean + yerr)
            print ('num trials for {}: {}, mean {}, ylower {}, yupper {}, mean rew {}'.format(vs_val, n, mean, ylower[-1], yupper[-1], mean_rew))
            x.append(vs_val)

        x = np.array(x)
        y = np.array(y)
        ylower = np.array(ylower)
        yupper = np.array(yupper)
        s_inds = np.argsort(x)
        x = x[s_inds]
        y = y[s_inds]
        ylower = ylower[s_inds]
        yupper = yupper[s_inds]
        linestyle = '-'
        if 'Abstract' in label or 'Abs'in label:
            linestyle = '-.'
        line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 2)
        #line, = plt.plot(x, y, label=label)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

    if plot_params['log_scale']:
        #ax.set_xscale('log')
        ax.set_yscale('log')
    if plot_params['x_range'] is not None:
        plt.xlim(plot_params['x_range'])
    if plot_params['y_range'] is not None:
        plt.ylim(plot_params['y_range'])

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def plot_training(data, batch, traj, file_name, plot_params, metric = 'mse'):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in sorted(data):
        label = method
        print ('method: {}'.format(method))
        if metric == 'mse':
            sub_data = np.array(data[method][batch][traj]['mse_training'])
        else:
            sub_data = np.array(data[method][batch][traj]['r_ests'])

        n = sub_data.shape[0]
        print ('{} trials {}'.format(method, n))
        y = np.mean(sub_data, axis = 0)
        std = np.std(sub_data, axis = 0)
        yerr = 1.96 * std / np.sqrt(float(n))
        ylower = y - yerr
        yupper = y + yerr

        tr_steps = sub_data.shape[1]
        x = [i * 1000 for i in range(1, tr_steps + 1)]       

        x = np.array(x)
        y = np.array(y)
        ylower = np.array(ylower)
        yupper = np.array(yupper)
        #x = np.arange(len(y))
        linestyle = '-'
        if 'Abstract' in label or 'Abs'in label:
            linestyle = '-.'
        line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 2)
        #line, = plt.plot(x, y, label=label)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

    if metric == 'rew':
        orac_est = data[method][batch][traj]['oracle_est']
        n = len(orac_est)
        y = np.mean(orac_est)
        std = np.std(orac_est)
        yerr = 1.96 * std / np.sqrt(float(n))
        ylower = y - yerr
        yupper = y + yerr

        x = np.array([i * 1000 for i in range(1, tr_steps + 1)])
        y = np.array([y for _ in range(1, tr_steps + 1)]) 
        ylower = np.array([ylower for _ in range(1, tr_steps + 1)]) 
        yupper = np.array([yupper for _ in range(1, tr_steps + 1)])
        line, = plt.plot(x, y, label='oracle', linewidth = 2)
        color = line.get_color()
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)
    
    if plot_params['log_scale']:
        #ax.set_xscale('log')
        ax.set_yscale('log')
    if plot_params['x_range'] is not None:
        plt.xlim(plot_params['x_range'])
    if plot_params['y_range'] is not None:
        plt.ylim(plot_params['y_range'])

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def plot_hp_sens(data, batch, traj_len, file_name, plot_params):
    
    #sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)
    
    #for method in data:
    for method in ['AbstractBestDICE', 'BestDICE']:
        label = method
        sub_data = data[method][batch][traj_len]
        xs = []
        ys = []
        ylowers = []
        yuppers = []
        for idx, hp in enumerate(sub_data):
            errors = sub_data[hp]['mse']
            x = hp[0]
            n = len(errors)
            y = np.mean(errors, axis = 0)
            std = np.std(errors, axis = 0)
            yerr = 1.96 * std / np.sqrt(float(n))
            ylower = y - yerr
            yupper = y + yerr
            xs.append(x)
            ys.append(y)
            ylowers.append(ylower)
            yuppers.append(yupper)
        xs = np.array(xs)
        ys = np.array(ys)
        ylowers = np.array(ylowers)
        yuppers = np.array(yuppers)
        s_inds = np.argsort(xs)
        xs = xs[s_inds]
        ys = ys[s_inds]
        ylowers = ylowers[s_inds]
        yuppers = yuppers[s_inds]
        linestyle = '-'
        if 'Abstract' in label or 'Abs'in label:
            linestyle = '-.'
        line, = plt.plot(xs, ys, label=label, linestyle = linestyle, linewidth = 2)
        #line, = plt.plot(x, y, label=label)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(xs, ylowers, yuppers, facecolor=color, alpha=alpha)

    if plot_params['log_scale']:
        #ax.set_xscale('log')
        ax.set_yscale('log')

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def plot_hp_sens_3d(data, method, batch, traj_len, file_name, plot_params, heatmap = False):
    
    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = plt.gca()

    data = data[method][batch][traj_len]

    qs = []
    ws = []
    errs = []
    for idx, hp in enumerate(data):
        #error = min(np.mean(data[hp]['mse']), 5)
        error = np.mean(data[hp]['mse'])
        print (hp, error)
        qs.append(hp[0])
        ws.append(hp[1])
        errs.append(error)

    qs = np.array(qs)
    ws = np.array(ws)
    errs = np.array(errs)
    if heatmap:
        errs_map = {}
        for idx, (i, j) in enumerate(zip(qs, ws)):
            errs_map[(i, j)] = errs[idx]
        qs = np.sort(np.array(list(set(qs))))
        ws = np.sort(np.array(list(set(ws))))
        data = np.zeros((len(qs), len(ws)))

        # qs becomes y axis
        for i in range(len(qs)):
            for j in range(len(ws)):
                pair = (qs[i], ws[j])
                if pair in errs_map:
                    data[i, j] = errs_map[pair]
                else:
                    data[i, j] = 0
        #cm = plt.get_cmap('Blues')
        cm = plt.get_cmap('YlOrRd')
        ms = ax.matshow(data, cmap = cm)

        ax.set_xticklabels([str(w) for w in ws])
        ax.set_yticklabels([str(q) for q in qs])

        ax.set_xticks(np.arange(data.shape[1]), minor = False)
        ax.set_yticks(np.arange(data.shape[0]), minor = False)
       
        cbar = fig.colorbar(ms, ticks = np.arange(np.min(data),np.max(data), (np.max(data) - np.min(data)) / 10))
        #cbar.ax.set_yticklabels(colorbar_ticklabels)

    else:
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(qs, ws, errs)
        ax.set_zlabel(plot_params['z_label'])

    ax.set_xlabel(plot_params['x_label'], labelpad = plot_params['axis_label_pad'])
    ax.set_ylabel(plot_params['y_label'], labelpad = plot_params['axis_label_pad'])

    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def plot_rank_mse(data, method, batch, traj_len, file_name, plot_params, plot_type = 'scatter'):
    
    sns.set_style("darkgrid")
    fig = plt.figure()
    if plot_type == 'scatter':
        f, axes = plt.subplots(nrows = 1, ncols = 2)
        f.set_size_inches(12, 10.0, forward=True)
    elif plot_type == 'heatmap':
        fig.set_size_inches(12, 10.)
        ax = plt.gca()
   
    data = data[method][batch][traj_len]

    q_ranks = []
    w_ranks = []
    errs = []
    for idx, hp in enumerate(data):
        error = np.mean(data[hp]['mse'])
        w_rank = round(np.mean(data[hp]['w_rank']), 3)
        q_rank = round(np.mean(data[hp]['q_rank']), 3)
        w_ranks.append(w_rank)
        q_ranks.append(q_rank)
        errs.append(error)

    q_ranks = np.array(q_ranks)
    w_ranks = np.array(w_ranks)
    errs = np.array(errs)
    if plot_type == 'scatter':
        axes[0].scatter(w_ranks, errs)
        axes[1].scatter(q_ranks, errs)
        axes[0].set_xlabel('W' + ' ' + plot_params['x_label'])
        axes[1].set_xlabel('Q' + ' ' + plot_params['x_label'])
        axes[0].set_ylabel(plot_params['y_label'])
        axes[1].set_ylabel(plot_params['y_label'])
    elif plot_type == 'heatmap':
        errs_map = {}
        for idx, (i, j) in enumerate(zip(q_ranks, w_ranks)):
            errs_map[(i, j)] = errs[idx]
        qs = np.sort(np.array(list(set(q_ranks))))
        ws = np.sort(np.array(list(set(w_ranks))))
        data = np.zeros((len(qs), len(ws)))

        # qs becomes y axis
        for i in range(len(qs)):
            for j in range(len(ws)):
                pair = (qs[i], ws[j])
                if pair in errs_map:
                    data[i, j] = errs_map[pair]
                else:
                    data[i, j] = 0
        cm = plt.get_cmap('Blues')
        ms = ax.matshow(data, cmap = cm)

        ax.set_xticklabels([str(w) for w in ws])
        ax.set_yticklabels([str(q) for q in qs])

        ax.set_xticks(np.arange(data.shape[1]), minor = False)
        ax.set_yticks(np.arange(data.shape[0]), minor = False)
        
        cbar = fig.colorbar(ms, ticks = np.arange(np.min(data),np.max(data), (np.max(data) - np.min(data)) / 10))
        #cbar = fig.colorbar(ms, ticks = np.arange(np.min(data),np.max(data), (2.5 - np.min(data)) / 10))
        ax.set_xlabel(plot_params['x_label'], labelpad = plot_params['axis_label_pad'])
        ax.set_ylabel(plot_params['y_label'], labelpad = plot_params['axis_label_pad'])

        plt.xticks(rotation = 90)

    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def plot_dens_scatter(data, batch, traj_len, file_name, plot_params, plot_type = 'scatter'):
    
    sns.set_style("darkgrid")
    fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)
    
    means = []
    for method in data:
        dens = data[method][batch][traj_len]['dens_est']
        mean = np.mean(dens, axis = 0)
        means.append(mean)
        print (method)
        print (mean)
    means = np.array(means)
    plt.scatter(means[0, :], means[1, :], s = 200)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
    fig.tight_layout()

    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def collect_data():

    data = {}

    for basename in os.listdir(FLAGS.result_directory):
        if '.npy' not in basename:
            continue
        f_name = os.path.join(FLAGS.result_directory, basename)
        try:
            results = np.load(f_name, allow_pickle = True).item()
        except Exception as e:
            if 'Tag' not in str(e):
                raise e
        # seed_574976_batch_50_traj_10_Qlr_0.001_Wlr_0.001_lamlr_0.001.npy
        '''
        method -> batch -> traj -> (qlr, wlr, lamlr) -> {e1, e2, ..., en}
        '''
        summary = np.load(f_name, allow_pickle = True).item()

        batch = summary['batch_size'] 
        traj_len = summary['traj_len']
        qlr = float(f_name.split('_')[-5])# summary['hp']['Q_lr']
        wlr = float(f_name.split('_')[-3]) #summary['hp']['W_lr']
        lamlr = float(f_name.split('_')[-1][:-4])#summary['hp']['lam_lr']
        hp = (qlr, wlr, lamlr)
        oracle_est = summary['oracle_est']
        results = summary['results']
        if 'Best Dice' in results:
            results['BestDice'] = results.pop('Best Dice')

        if 'BestDice' in results:
            results['BestDICE'] = results.pop('BestDice')

        if 'AbsBestDice' in results:
            results['AbstractBestDICE'] = results.pop('AbsBestDice')
       
        algos = summary['results'].keys()

        if qlr == 0 or wlr == 0:
            continue

        #results_AntUMaze_gan_mdp_0_pi_0_mix_0.5_2/env_AntUMaze_exp_gan_seed_848550_batch_5_traj_500_mix_0.5_Qlr_0.0007_Wlr_0.0001_lamlr_0.001.npy
        for algo in algos:
            #if 'True' not in algo:
            #    continue
            if algo not in data:
                data[algo] = {}
            if batch not in data[algo]:
                data[algo][batch] = {}
            if traj_len not in data[algo][batch]:
                data[algo][batch][traj_len] = {}
            if hp not in data[algo][batch][traj_len]:
                data[algo][batch][traj_len][hp] = {
                    'mse': [],
                    'mse_training': [],
                    'oracle_est': [],
                    'r_ests': [],
                    'w_rank': [],
                    'q_rank': [],
                    'r_est': [],
                    'dens_est': []
                }

            data[algo][batch][traj_len][hp]['mse'].append(results[algo]['mse'] if 'mse' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['mse_training'].append(results[algo]['mse_training'] if 'mse_training' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['r_est'].append(results[algo]['r_est'] if 'r_est' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['r_ests'].append(results[algo]['r_ests'] if 'r_ests' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['oracle_est'].append(oracle_est)
            data[algo][batch][traj_len][hp]['w_rank'].append(results[algo]['w_rank'] if 'w_rank' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['q_rank'].append(results[algo]['q_rank'] if 'q_rank' in results[algo] else 0)
            data[algo][batch][traj_len][hp]['dens_est'].append(results[algo]['dens_est'] if 'dens_est' in results[algo] else 0)
            
            
    return data

def best_hp(data):
    best_data = {}
    for method in data:
        best_data[method] = {}
        for batch in data[method]:
            best_data[method][batch] = {}
            for traj_len in data[method][batch]:
                best_data[method][batch][traj_len] = {}
                min_err = float('inf')
                best_hp = -1
                for hp in data[method][batch][traj_len]:
                    errs = np.array(data[method][batch][traj_len][hp]['mse'])
                    mean = np.mean(errs)
                    if mean < min_err:
                        min_err = mean
                        best_hp = hp
                print ('method {} batch {} traj {} best hp {}'.format(method, batch, traj_len, best_hp))
                if best_hp == -1:
                    best_hp = hp
                
                best_data[method][batch][traj_len]['mse'] = data[method][batch][traj_len][best_hp]['mse']
                best_data[method][batch][traj_len]['mse_training'] = data[method][batch][traj_len][best_hp]['mse_training']
                best_data[method][batch][traj_len]['r_est'] = data[method][batch][traj_len][best_hp]['r_est']
                best_data[method][batch][traj_len]['r_ests'] = data[method][batch][traj_len][best_hp]['r_ests']
                best_data[method][batch][traj_len]['oracle_est'] = data[method][batch][traj_len][best_hp]['oracle_est']
                best_data[method][batch][traj_len]['w_rank'] = data[method][batch][traj_len][best_hp]['w_rank']
                best_data[method][batch][traj_len]['q_rank'] = data[method][batch][traj_len][best_hp]['q_rank']
                best_data[method][batch][traj_len]['dens_est'] = data[method][batch][traj_len][best_hp]['dens_est']

    return best_data 

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = collect_data() 
    best_hp_data = best_hp(data)
    nice_fonts = {
        "pgf.texsystem": "pdflatex",
        # Use LaTex to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    #plt.figure()
    plt.style.use('seaborn')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams.update(nice_fonts)
    sns.set(rc = nice_fonts)

    plot_params = {'bfont': 50,
               'lfont': 43,
               'tfont': 50,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               'y_label': r'(relative) MSE($\rho(\pi_e)$)',
               #'y_label': r'MSE($\rho(\pi_e)$)',
               #'y_label': '(relative) MSE',
               'shade_error': True,
               'x_mult': 1,
               'axis_label_pad': 15}

    fname = '{}_'.format(FLAGS.domain)

    if FLAGS.plot_type == 'data':
        #plot_params['log_scale'] = False 
        plot_params['x_label'] = 'Trajectory Length' if FLAGS.vs == 'traj' else 'Batch Size (\# of Trajectories)'
        fname += 'vs_{}_other_fixed_{}'.format(FLAGS.vs, FLAGS.other_fixed_val)
        plot_vs(best_hp_data, FLAGS.vs, FLAGS.other_fixed_val, fname, plot_params)
    elif FLAGS.plot_type == 'training':
        plot_params['x_label'] = 'training steps'
        #plot_params['log_scale'] = False 
        if FLAGS.tr_metric == 'rew':
           plot_params['y_label'] = r'$\rho(\pi_e)$'
           #plot_params['y_label'] = 'rew'
           plot_params['log_scale'] = False 
        fname += '{}_training_b_{}_t_{}'.format(FLAGS.tr_metric, FLAGS.batch, FLAGS.traj)
        plot_training(best_hp_data, FLAGS.batch, FLAGS.traj, fname, plot_params, metric = FLAGS.tr_metric)
    elif FLAGS.plot_type == 'hp_sensitivity_3d':
        #plot_params['x_label'] = 'W' 
        plot_params['x_label'] = r'$\alpha_\zeta$' 
        plot_params['y_label'] = r'$\alpha_\nu$'
        #plot_params['y_label'] = 'Q' 
        plot_params['z_label'] = 'MSE'
        fname += 'hp_sens_3d_{}_b_{}_t_{}'.format(FLAGS.method, FLAGS.batch, FLAGS.traj)
        plot_hp_sens_3d(data, FLAGS.method, FLAGS.batch, FLAGS.traj, fname, plot_params, heatmap = True)
    elif FLAGS.plot_type == 'rank_mse_scat':
        plot_params['x_label'] = 'Rank' 
        plot_params['y_label'] = 'MSE'
        fname += 'rank_mse_scat_{}_b_{}_t_{}'.format(FLAGS.method, FLAGS.batch, FLAGS.traj)
        plot_rank_mse(data, FLAGS.method, FLAGS.batch, FLAGS.traj, fname, plot_params, plot_type = 'scatter')
    elif FLAGS.plot_type == 'rank_mse_heat':
        plot_params['x_label'] = 'W' 
        plot_params['y_label'] = 'Q'
        fname += 'rank_mse_scat_{}_b_{}_t_{}'.format(FLAGS.method, FLAGS.batch, FLAGS.traj)
        plot_rank_mse(data, FLAGS.method, FLAGS.batch, FLAGS.traj, fname, plot_params, plot_type = 'heatmap')
    elif FLAGS.plot_type == 'dens_scatter':
        fname += 'dens_scat_b_{}_t_{}'.format(FLAGS.batch, FLAGS.traj)
        plot_params['x_label'] = r'$d_{\pi^\phi_e}(s^\phi,a)$' 
        plot_params['y_label'] = r'$\hat{d}_{\pi^\phi_e}(s^\phi,a)$'
        plot_dens_scatter(best_hp_data, FLAGS.batch, FLAGS.traj, fname, plot_params, plot_type = 'scatter')
    elif FLAGS.plot_type == 'hp_sensitivity':
        plot_params['log_scale'] = False
        plot_params['x_label'] = r'$\alpha_\zeta=\alpha_\nu$' 
        plot_params['y_label'] = r'(relative) MSE$(\rho(\pi_e))$'
        fname += 'hp_sens_bowl_b_{}_t_{}'.format(FLAGS.batch, FLAGS.traj)
        plot_hp_sens(data, FLAGS.batch, FLAGS.traj, fname, plot_params)
        

    
if __name__ == '__main__':
    main()


