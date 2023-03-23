import os 
import pandas as pd 
import numpy as np

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))

# --------- TI task ---------- #

def mk_tiTask(mode, n_items=9):

    # create stimulus sequence
    items = [ (i, i+1) 
            for i in range(n_items-1)]
    items = items + list(reversed(items))
    if mode == 'jumps': np.random.shuffle(items)

    items = np.vstack(items)
    task ={
        'stim1': list(items[:, 0]),
        'stim2': list(items[:, 1]),
        'trial': list(range(2*(n_items-1)))
    }
    
    return pd.DataFrame.from_dict(task)

# --------- Particle Filter ---------- #

class particleFilter:

    def __init__(self, params, n_items=9, seed=2023):
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)
        self._load_params(params)
        self._init_filters()

    def _load_params(self, params):
        self.sig0  = params[0]
        self.sigd  = params[1]
        self.beta  = params[2]
        self.N     = params[3]
        self.alpha = params[4]

    def _init_filters(self):
        # dims: (num of filters, num of items)
        self.filters = self.sig0*self.rng.randn(self.N, self.n_items)
        # uniform weights: w0 = 1/N
        self.weights = np.ones([self.N]) / self.N

    def reweight(self, y):
        
        # unpack the current observation
        i, j = y 
        diff = self.filters[:, j] - self.filters[:, i]
        # local updating
        lcl = 1 / (1 + np.exp(-self.beta*(diff)))
        # global updating
        glb = 1*(diff > 0)
        # like
        g_y1v = self.alpha*glb + lcl
        # reweight and normalized 
        weights = g_y1v * self.weights
        self.weights = weights / weights.sum()

    def resampling(self):

        # sampling with replacement
        sel_idx = self.rng.choice(self.N, size=self.N, 
                    replace=True, p=self.weights.reshape([-1]))
        self.filters = self.filters[sel_idx, :]
        # reset weights to w0 = 1/N
        self.weights = np.ones([self.N]) / self.N

    def propagation(self):

        # random drft the particles 
        self.filters += self.sigd*self.rng.randn(self.N, self.n_items)

    def step(self, y):

        self.reweight(y)
        self.resampling()
        self.propagation()

    
