# Taken from: https://github.com/Lab-ANT/Time2State/

import numpy as np
from pyhsmm import distributions
from pyhsmm import models
from pyhsmm.util.text import progprint_xrange


class HDP_HSMM:

    def __init__(self, alpha, beta, n_iter=20):
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit_transform(self, X):
        data = X
        # Set the weak limit truncation level
        Nmax = 60
        # and some hyperparameters
        obs_dim = data.shape[1]
        obs_hypparams = {'mu_0': np.zeros(obs_dim),
                         'sigma_0': np.eye(obs_dim),
                         'kappa_0': 0.25,
                         'nu_0': obs_dim + 2}
        dur_hypparams = {'alpha_0': self.alpha,
                         'beta_0': self.beta}

        obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = models.WeakLimitHDPHSMM(
            alpha=6., gamma=6.,  # these can matter; see concentration-resampling.py
            init_state_concentration=6.,  # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns
        )
        posteriormodel.add_data(data, trunc=600)  # duration truncation speeds things up when it's possible

        for idx in progprint_xrange(self.n_iter):
            posteriormodel.resample_model()

        return posteriormodel.stateseqs[0]
