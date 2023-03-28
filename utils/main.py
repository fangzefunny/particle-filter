import os 
import pandas as pd 
import numpy as np

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))
eps_ = 1e-12

# --------- TI task ---------- #

def mk_tiTask(mode, n_items=9, seed=2023):

    # random generator
    rng = np.random.RandomState(seed)

    # create the training stimulus sequence
    # "In both conditions the sequence of 8 
    #  premises waspresented twice in each  
    #  training block and the  direction of 
    #  the sequences (forward or backward 
    #  through the hierarchy)alternated across 
    #  blocks, with the direction in the first 
    #  blockrandomized for each participant."
    items = [ (i, i+1) 
            for i in range(n_items-1)]
    if mode == 'jumps': rng.shuffle(items)
    train = items + list(reversed(items)) + items + list(reversed(items))

    # create the test sequence
    lst  = list(range(n_items)) * 5
    rng.shuffle(lst)
    test = [(lst[i], lst[i+1]) for i in range((len(lst)-1))]
    gaps = [np.abs(i-j) for i, j in test]

    train = np.vstack(train)
    test  = np.vstack(test)
    task ={
        'stim1': list(train[:, 0])+list(test[:, 0]),
        'stim2': list(train[:, 1])+list(test[:, 1]),
        'trial': list(range(len(train)+len(lst)-1)),
        'type' : list(['train']*len(train))+list(['test']*(len(lst)-1)),
        'gap'  : [0]*len(train)+gaps
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

    def like(self, obs):

        # unpack the current observation
        # i < j
        i, j = obs 
        diff = self.filters[:, j] - self.filters[:, i]
        # local updating
        lcl = 1 / (1 + np.exp(-self.beta*(diff)))
        # global updating
        glb = 1*(diff > 0)
        # like
        return self.alpha*glb + lcl
    
    def eval(self, obs):

        like1 = self.like(obs).mean()
        like2 = self.like(list(reversed(obs))).mean()
        return like1 / (like1 + like2)

    def reweight(self, obs):
        
        # get the likelihood 
        g_y1v = self.like(obs)

        # reweight and normalized 
        weights =  g_y1v * self.weights
        weights = weights / weights.sum()
        self.weights = weights

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

    def step(self, obs):

        self.reweight(obs)
        self.resampling()
        self.propagation()


class particleFilter2D(particleFilter):

    def __init__(self, params, n_items=9, seed=2023):
        super().__init__(params, n_items, seed)

    def _init_filters(self):
        # dims: (num of filters, num of items, num of items)
        self.filters = self.sig0*self.rng.randn(
            self.N, self.n_items, self.n_items)
        # uniform weights: w0 = 1/N
        self.weights = np.ones([self.N]) / self.N

    def reweight(self, obs, rel):
        
        # unpack the current observation
        (xi, yi), (xj, yj) = obs
        # pos val if the filter directioin agreed with the real direction
        # e.g. 
        # real: xi < xj, filter: f(xi) - f(xj) < 0; x_diff > 0 
        # real: xi > xj, filter: f(xi) - f(xj) < 0; x_diff < 0 
        # real: xi > xj, filter: f(xi) - f(xj) > 0; x_diff > 0 
        # real: xi < xj, filter: f(xi) - f(xj) > 0; x_diff < 0 
        diff = (self.filters[:, xi, yi]-
                self.filters[:, xj, yj])* rel.shape([-1, 2])
        # local updating
        lcl = 1 / (1 + np.exp(-self.beta*(diff.sum(axis=(1, 2)))))
        # global updating
        # x_diff > 0, y_diff > 0; glb = .5+ (1+1)/4  = 1
        # x_diff < 0, y_diff > 0; glb = .5+ (-1+1)/4 = .5 
        # x_diff > 0, y_diff < 0; glb = .5+ (1+-1)/4 = .5 
        # x_diff < 0, y_diff < 0; glb = .5+ (-1-1)/4 = 0
        glb = 1 * (.5+(np.sign(diff).sum(axis=(1,2)))/8)
        # like
        g_y1v = self.alpha*glb + lcl
        # reweight and normalized 
        weights = g_y1v * self.weights
        self.weights = weights / weights.sum()

    def resampling(self):

        # sampling with replacement
        sel_idx = self.rng.choice(self.N, size=self.N, 
                    replace=True, p=self.weights.reshape([-1]))
        self.filters = self.filters[sel_idx, :, :]
        # reset weights to w0 = 1/N
        self.weights = np.ones([self.N]) / self.N

    def propagation(self):

        # random drft the particles 
        self.filters += self.sigd*self.rng.randn(
            self.N, self.n_items, self.n_items)

    def step(self, obs, rel):

        self.reweight(obs, rel)
        self.resampling()
        self.propagation()

    
if __name__ == '__main__':

    task = mk_tiTask('chain')

