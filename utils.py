import numpy as np
import pdb
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import warnings
warnings.filterwarnings("error")

def collect_data_discrete(env, policy, num_trajectory, truncated_horizon, gamma = None):
    if not gamma:
        gamma = 1.
    phi = env.phi
    paths = []
    total_reward = 0.0
    densities = np.zeros((env.n_state, truncated_horizon))
    frequency = np.zeros(env.n_state)
    for i_trajectory in range(num_trajectory):
        path = {}
        path['obs'] = []
        path['acts'] = []
        path['rews'] = []
        path['nobs'] = []
        state = env.reset()
        sasr = []
        accum_gamma = np.ones(env.n_state)
        for i_t in range(truncated_horizon):
            action = policy(state)
            #p_action = policy[state, :]
            #action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward, done, _ = env.step(action)
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            #sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            densities[state, i_t] += 1
            total_reward += reward
            state = next_state
            if done:
                break
        paths.append(path)

    gammas = np.array([gamma ** i for i in range(truncated_horizon)])
    d_sum = np.sum(densities, axis = 0)
    densities = np.divide(densities, d_sum, out=np.zeros_like(densities), where = d_sum != 0)
    disc_densities = np.dot(densities, gammas)
    final_densities = (disc_densities / np.sum(gammas))
    return paths, frequency, total_reward / (num_trajectory * truncated_horizon), final_densities

def collect_data(env, policy, num_trajectory, truncated_horizon):
    paths = []
    num_samples = 0
    total_reward = 0.0
    for i_trajectory in range(num_trajectory):
        path = {}
        path['obs'] = []
        path['nobs'] = []
        path['acts'] = []
        path['rews'] = []
        state = env.reset()
        sasr = []
        for i_t in range(truncated_horizon):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            #sasr.append((state, action, next_state, reward))
            total_reward += reward
            state = next_state
            if done:
                break
        paths.append(path)
        num_samples += len(paths[-1]['obs'])
    return paths, total_reward / num_samples#(num_trajectory * truncated_horizon)

def format_data_discrete(mdp, data, pie, gamma, abs_pie, pib = None):
    batch_data = data['data']

    s = []
    sprime = []
    sa = []
    sprime_a = []
    abs_s = []
    abs_sprime = []
    abs_sa = []
    abs_sprime_a = []
    abs_sprime_a_pract = []
    rewards = []
    gammas = []
    trans_tups = []
    abs_trans_tups = []
    policy_ratios = []

    pg = np.zeros((mdp.n_state, mdp.n_action, mdp.n_state))
    pa = np.zeros((mdp.n_abs_s, mdp.n_action, mdp.n_abs_s))
    norms = np.zeros(mdp.n_abs_s)
    for idx in range(len(batch_data)):
        path = batch_data[idx]
        obs = path['obs']
        nobs = path['nobs']
        acts = path['acts']
        rews = path['rews']
        accum_gamma = 1.
        for t in range(len(obs)):
            o = obs[t]
            no = nobs[t]
            a_pib = acts[t]
            r = rews[t]
 
            pg[o, a_pib, no] += 1
            pa[mdp.phi(o), a_pib, mdp.phi(no)] += 1
            norms[mdp.phi(o)] += 1

            # used by: ground (w, Q), pract abs (Q -- Bellman)
            sa_b = mdp.get_feature_vector(o, a_pib)
            exp_n_sa_e = np.zeros(mdp.n_sa)

            # used by: pure abstract (w, Q), pract abs (w -- Bellman), pract abs (w, Q -- non-Bellman)
            abs_sa_b = mdp.get_feature_vector(mdp.phi(o), a_pib, abst = True)
            exp_abs_n_sa_e = np.zeros(mdp.n_abs_sa)
            exp_abs_n_sa_e_pract = np.zeros(mdp.n_abs_sa)

            # compute value function following \pi_e from next state
            for a in range(mdp.n_action):
                # used by: ground (Q), pract abs (Q -- Bellman)
                exp_n_sa_e += pie.get_prob(no, a) * mdp.get_feature_vector(no, a)
                # querying abstract pie
                # used by: pure abstract (Q)
                if abs_pie:
                    exp_abs_n_sa_e += abs_pie.get_prob(mdp.phi(no), a) * mdp.get_feature_vector(mdp.phi(no), a, abst = True)
                # querying ground pie
                # used by: pract abs (Q -- non-Bellman)
                exp_abs_n_sa_e_pract += pie.get_prob(no, a) * mdp.get_feature_vector(mdp.phi(no), a, abst = True)

            ratio = -1
            if pib:
                ratio = pie.get_prob(o, a_pib) / pib.get_prob(o, a_pib)
            policy_ratios.append(ratio)

            trans_tups.append((o, a_pib, r, no))
            abs_trans_tups.append((mdp.phi(o), a_pib, r, mdp.phi(no)))
            s.append(mdp.get_s_feature_vector(o))
            sprime.append(mdp.get_s_feature_vector(no))
            sa.append(sa_b)
            sprime_a.append(exp_n_sa_e)
            abs_s.append(mdp.get_s_feature_vector(mdp.phi(o), abst = True))
            abs_sprime.append(mdp.get_s_feature_vector(mdp.phi(no), abst = True))
            abs_sa.append(abs_sa_b)
            abs_sprime_a.append(exp_abs_n_sa_e)
            abs_sprime_a_pract.append(exp_abs_n_sa_e_pract)
            rewards.append(r)
            gammas.append(accum_gamma)
            accum_gamma *= gamma

    norms = norms + 1e-10
    for i in range(mdp.n_abs_s):
        pa[i] = pa[i] / (np.sum(pa[i], axis =1)[:, None] + 1e-10)

    init_states = []
    exp_init_sa_e = np.zeros(mdp.n_sa)
    exp_abs_init_sa_e = np.zeros(mdp.n_abs_sa)
    densities = data['initial_states']
    for d in densities:
        for a in range(mdp.n_action):
            # used by: ground (Q), pract abs (Q -- Bellman)
            exp_init_sa_e += densities[d] * pie.get_prob(d, a) * mdp.get_feature_vector(d, a)
            # used by: pure abstract (Q), pract abs (Q -- non-Bellman)
            # rhs of equality works with just querying ground query due to lemma
            exp_abs_init_sa_e += densities[d] * pie.get_prob(d, a) * mdp.get_feature_vector(mdp.phi(d), a, abst = True)
        init_states.append(d)
    data = {
        'state_b': np.array(s),
        'next_state_b': np.array(sprime),
        'state_b_act_b': np.array(sa),
        'exp_next_state_b_act_e': np.array(sprime_a),
        'abs_state_b': np.array(abs_s),
        'abs_next_state_b': np.array(abs_sprime),
        'abs_state_b_act_b': np.array(abs_sa),
        'exp_abs_next_state_b_act_e': np.array(abs_sprime_a),
        'exp_abs_next_state_b_act_e_pract': np.array(abs_sprime_a_pract),
        'rewards': np.array(rewards),
        'gammas': np.array(gammas),
        'init_state_act_e': exp_init_sa_e,
        'abs_init_state_act_e': exp_abs_init_sa_e,
        'num_samples': len(sa),
        'ground_trans': trans_tups,
        'abs_trans': abs_trans_tups,
        'init_states': densities,
        'policy_ratios': np.array(policy_ratios)
    }
    return data 

def format_data_new(data, phi, gamma):
    g_data = data['ground_data']

    s = []
    sa = []
    sprime = []
    abs_s = []
    abs_sa = []
    abs_sprime = []
    rewards = []
    gammas = []
    for idx in range(len(g_data)):
        path = g_data[idx]
        obs = path['obs']
        nobs = path['nobs']
        acts = path['acts']
        rews = path['rews']
        accum_gamma = 1.
        for t in range(len(obs)):
            o = obs[t]
            no = nobs[t]
            a_pib = acts[t]
            if not (isinstance(a_pib, list) or isinstance(a_pib, np.ndarray)):
                a_pib = [a_pib]
            r = rews[t]
            s.append(o)
            sa.append(np.concatenate([o, a_pib]))
            sprime.append(no)
            abs_s.append(phi(o))
            abs_sa.append(np.concatenate([phi(o), a_pib]))
            abs_sprime.append(phi(no))
            rewards.append(r)
            gammas.append(accum_gamma)
            accum_gamma *= gamma

    abs_init_states = []
    for d in data['initial_states']:
        abs_init_states.append(phi(d))

    data = {
        'state_b': np.array(s),
        'state_b_act_b': np.array(sa),
        'abs_state_b': np.array(abs_s),
        'abs_state_b_act_b': np.array(abs_sa),
        'next_state_b': np.array(sprime),
        'abs_next_state_b': np.array(abs_sprime),
        'rewards': np.array(rewards),
        'gammas': np.array(gammas),
        'init_state': data['initial_states'],
        'abs_init_state': np.array(abs_init_states),
        'num_samples': len(sa)
    }
    return data 

def compute_Q_func(mdp, batch_data, gamma):
    import copy
    q = np.zeros((mdp.n_state, mdp.n_action))
    prev_q = np.zeros_like(q)
    alpha = 0.005

    for itr in range(10000):
        for idx in range(len(batch_data)):
            path = batch_data[idx]
            obs = path['obs']
            nobs = path['nobs']
            acts = path['acts']
            rews = path['rews']
            for t in range(len(obs)):
                o = obs[t]
                no = nobs[t]
                a = acts[t]
                r = rews[t]
                if t < len(obs) - 1:
                    na = acts[t + 1]
                    q[o, a] = q[o, a] + alpha * (r + gamma  * q[no, na] - q[o, a])
                else:
                    q[o, a] = q[o, a] + alpha * (r - q[o, a])

        if ((itr + 1) % 50 == 0):
            diff = np.abs(q - prev_q)
            thresh = (diff <= 1e-10)
            count = np.count_nonzero(thresh)

            if count == mdp.n_state * mdp.n_action:
                print ('converged, done, itr {}'.format(itr + 1))
                break
        prev_q = copy.deepcopy(q)

    q_func = q
    pdb.set_trace()
    return q_func

def compute_dsa(mdp, state_dens, pi):
    dsa = np.zeros((mdp.n_state, mdp.n_action))

    for s in range(mdp.n_state):
        for a in range(mdp.n_action):
            dsa[s, a] = state_dens[s] * pi.get_prob(s, a)
    dsa = np.matrix.flatten(dsa)
    return dsa

def compute_abs_densities(g_dens, pi, mdp, state_only = False):
    if state_only:
        abs_dens = np.zeros((mdp.n_abs_s))
    else:
        abs_dens = np.zeros((mdp.n_abs_s, mdp.n_action))
    for i in range(len(g_dens)):
        if state_only:
            abs_s = mdp.phi(i)
            abs_dens[abs_s] += g_dens[i]
        else:
            for a in range(mdp.n_action):
                abs_s = mdp.phi(i)
                abs_dens[abs_s, a] += g_dens[i] * pi.get_prob(i, a)
    abs_dens = np.matrix.flatten(abs_dens)
    return abs_dens

def compute_abs_model(mdp, g_densities):
    abs_P = np.zeros((mdp.n_abs_s, mdp.n_action, mdp.n_abs_s))
    norms = np.zeros(mdp.n_abs_s)
    g_P = mdp.P
    for s in range(mdp.n_state):
        for a in range(mdp.n_action):
            for ns in range(mdp.n_state):
                val = g_densities[s] * g_P[s, a, ns]
                abs_P[mdp.phi(s), a, mdp.phi(ns)] += val
        norms[mdp.phi(s)] += g_densities[s]
    norms = norms + 1e-10
    for i in range(mdp.n_abs_s):
        abs_P[i] = abs_P[i] / norms[i]
    print (abs_P)
    return abs_P

def normalize_data(data):
    # assumed format:
    # M is n * d matrix, where n is the number of samples, d is the dimension of each vector
    # normalizes each feature to be between 0 and 1

    data = np.array(data)
    #return data
    scaler = preprocessing.StandardScaler()
    #scaler = MinMaxScaler()
    scaler.fit(data)
    #data_min = np.min(data, axis = 0)
    #data_max = np.max(data, axis = 0)
    #normalized = (data - data_min) / (data_max - data_min)
    normalized = scaler.transform(data)
    return normalized

def get_MSE(true_val, pred_vals):
    sq_error = np.square(np.array(true_val) - np.array(pred_vals))
    res = get_CI(sq_error)
    return res

# statistics/visualization related
def get_CI(data, confidence = 0.95):

    if (np.array(data) == None).all():
        return {}
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    stats = {}
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    err = z * (std / np.sqrt(n))
    lower = mean - z * (std / np.sqrt(n))
    upper = mean + z * (std / np.sqrt(n))
    stats = {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'err': err,
        'max': np.max(data),
        'min': np.min(data)
    }
    return stats

def plot(xs, *args):
    alpha = 0.25
    for arg in args:
        means = np.array([stat['mean'] for stat in arg['stats']])
        errs = np.array([stat['err'] for stat in arg['stats']])

        line, = plt.plot(xs, means, label = arg['name'])
        color = line.get_color()
        plt.fill_between(xs, means - errs, means + errs, alpha=alpha, facecolor=color)
    plt.legend(loc = 'best')
    plt.show()

def heat_map_discrete(f, env, filename):
    p_matrix = np.zeros([env.n_bins_y, env.n_bins_x], dtype = np.float32)
    for state in range(env.n_state):
        x, y = env.state_decoding(state, actual_grid = False)
        x, y = int(x), int(y)
        p_matrix[y, x] = f[state]
    p_matrix = p_matrix / np.sum(p_matrix)
   
    ax = sns.heatmap(p_matrix, cmap="YlGnBu")#, vmin = 0.0, vmax = 0.07)
    #ax.set_yticklabels(ax.get_yticks(), rotation = 45)
    ax.invert_yaxis()
    plt.savefig(filename)
    plt.close()

def heat_map(env, SASs, filename):

    xs = []
    ys = []
    for SAS in SASs:
        x = [] 
        y = []
        for path in SAS:
            obs = path['obs']
            for o in obs:
                x.append(o[0])
                y.append(o[1])
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    #heatmap, xedges, yedges = np.histogram2d(x, y, bins = (10 * env.n, 10 * env.m), range = [[0, env.n], [0, env.m]], density = True)
    #heatmap, xedges, yedges = np.histogram2d(x, y, bins = (2, env.m), range = [[0, env.n], [0, env.m]], density = True)
    #extent = [0, env.n, 0, env.m]
    '''
    fg = sns.FacetGrid(df, col="case_type", col_wrap=2, sharey=False)
    #create common colorbar axis
    cax = fg.fig.add_axes([.92, .12, .02, .8])
    #map colorbar to colorbar axis with common vmin/vmax values
    fg.map(sns.histplot,"date", "range", stat="count", bins=bin_nr, vmin=vmin_all, vmax=vmax_all, cbar=True, cbar_ax=cax, data=df)
    #prevent overlap
    fg.fig.subplots_adjust(right=.9)
    fg.set_xticklabels(rotation=30)
    '''
    nice_fonts = {
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    mpl.rcParams.update(nice_fonts)
    
    cm = plt.get_cmap('Blues')
    plt.clf()
    bins = 100
    vmin = float('inf')
    vmax = -float('inf')
    #plt.imshow(heatmap.T, extent=extent, origin='lower')
    fig, ax = plt.subplots(1, len(SASs), figsize = (30, 15), sharex = True, sharey = True)
    for i in range(len(SASs)):
        h = sns.histplot(x = xs[i], y = ys[i], cbar = True, ax = ax[i], stat = 'probability', palette = cm, bins = bins)
        pdb.set_trace()
        c = plt.Circle((1., 8.8), 0.5, fill = False, linewidth = 3)  
        #h = np.histogram2d(x = xs[i], y = ys[i], bins = bins, density = True)
        #vmin = min(vmin, np.min(h[0]))
        #vmax = max(vmax, np.max(h[0]))
        ax[i].add_artist(c)
        ax[i].set(xlabel = 'width', ylabel = 'length')
    #ax.set_xlim(-0.5, 0.5)

    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
      
    # creating ScalarMappable
    #sm = plt.cm.ScalarMappable(norm=norm, cmap = cm)
    #sm.set_array([])
      
    #plt.colorbar(sm)
    #plt.colorbar()
    plt.savefig(filename)
    plt.close()
