import os
import numpy as np 
import pandas as pd 

import multiprocessing as mp 

import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.main import *
from utils.viz import viz
viz.get_style()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{pth}/data'): os.mkdir(f'{pth}/data')
if not os.path.exists(f'{pth}/figures'): os.mkdir(f'{pth}/figures')

lst = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


# ------- parallel -------- #

def get_pool(n_cores):
    n_cores = n_cores if n_cores else int(mp.cpu_count()*.7) 
    print(f'    Using {n_cores} parallel CPU cores\n ')
    return mp.Pool(n_cores)


# ------- simulation ------ #

def sim_block(params, mode, seed):

    # instantiate a task
    task  = mk_tiTask(mode, seed=seed)
    model = particleFilter(params, seed=seed*3)

    # predictive data
    col = ['acc']
    init_mat = np.zeros([task.shape[0], len(col)]) + np.nan 
    pred_data = pd.DataFrame(init_mat, columns=col)

    # for each trial, we do 
    for t, row in task.iterrows():

        # obtain the input 
        # make s1 < s2 
        obs = np.sort([row['stim1'], row['stim2']]).tolist()
        typ = row['type']

        # predict
        acc = model.eval(obs)
        pred_data.loc[t, 'acc'] = acc

        # update the model if 
        # in the training stage 
        if typ == 'train':
            model.step(obs)

    return pd.concat([task, pred_data], axis=1)

def sim_parallel(pool, params, mode, n_sim=500, seed=1233):
    
    res = [pool.apply_async(sim_block,
                    args=(params, mode, seed+i))
                    for i in range(n_sim)]
    
    sim_data = pd.concat([p.get() for p in res], 
                         axis=0, ignore_index=True)
    
    sim_data.to_csv(f'{pth}/data/sim_data-{mode}.csv')

# ------ Visualization -------- #

def comp_curves():

    # load data

    sim_data = {}
    modes  = ['chains', 'jumps']
    colors = [viz.Red, viz.Blue]
    for m in modes:
        sdata = pd.read_csv(f'{pth}/data/sim_data-{m}.csv')
        sim_data[m] = sdata.query('type=="test" & gap>0').reset_index()
        
    # plot the figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i, m in enumerate(modes):
        sns.lineplot(x='gap', y='acc', data=sim_data[m],
                    marker='s', lw=2, markersize=6, color=colors[i],
                    label=m, ax=ax)
    ax.set_ylim([.5, 1])
    ax.set_xticks(list(range(1, 9)))
    ax.set_xticklabels(list(range(1, 9)))
    ax.set_xlabel('Distance')
    ax.set_ylabel('% Correct')
    
    fig.tight_layout()  

    plt.savefig(f'{pth}/figures/test_acc.png', dpi=300)


if __name__ == '__main__':

    # get multiple pool
    pool = get_pool(50)

    # the model parametters 
    # sig0: the sd of the gaussian distribution that initiate the filters
    # sigd: the sd of the guassian distribution during the drift 
    # beta: the inverse temperature for the local update
    # N:    the number of filters
    # alpha: the weight between global and local updating
    params = [.2, 1, 2, 200, 1]
    for m in ['chains', 'jumps']:
        sim_parallel(pool, params, m)

    #viz 
    comp_curves()

    

        

        


    