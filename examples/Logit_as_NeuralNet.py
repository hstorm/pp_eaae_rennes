# %%
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.special import softmax

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


import flax.linen as nn
import jax
from jax import device_put, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive

from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO, autoguide, init_to_feasible
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO

from numpyro.contrib.module import flax_module, random_flax_module
from numpyro.infer import SVI, TraceMeanField_ELBO

# if "SVG" in os.environ:
#     %config InlineBackend.figure_formats = ["svg"]
az.style.use("arviz-darkgrid")
# numpyro.set_platform("cpu")
numpyro.set_platform("gpu")
# numpyro.set_host_device_count(2)

# %%
rng_key = random.PRNGKey(1)

# %%
# Useful source: https://omarfsosa.github.io/bayesian_nn

# %%
# Define a Flax NN model class which can be used within the DGP
from typing import Sequence
class MLP(nn.Module):
    """
    Flexible MLP module, allowing different number of layer and layer size, as
    well as dropout.
    # Run and inspect model
    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    model = MLP([12, 8, 4, 1], [0.0, 0.2, 0.3])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch,is_training=False)
    output = model.apply(variables, batch,is_training=True, rngs={'dropout': dropout_key})
    print(output.shape)  # (32, 1)

    # inspect model
    jax.tree_util.tree_map(jnp.shape, variables)
    """  
    
    lst_layer: Sequence[int]
    dropout_rates: Sequence[float]

    @nn.compact
    def __call__(self, x, is_training:bool):
        assert len(self.lst_layer) == len(self.dropout_rates) + 1
        
        for iLayer in range(0,len(self.lst_layer[:-1])):
            # x = nn.tanh(nn.Dense(self.lst_layer[iLayer])(x))
            # x = nn.relu(nn.Dense(self.lst_layer[iLayer])(x))
            x = nn.leaky_relu(nn.Dense(self.lst_layer[iLayer])(x))
        
            if self.dropout_rates[iLayer] > 0.0:
                x = nn.Dropout(self.dropout_rates[iLayer], 
                    deterministic=not is_training)(x)
        
        # x = nn.BatchNorm(
        #     use_bias=False,
        #     use_scale=False,
        #     momentum=0.9,
        #     use_running_average=not is_training,
        # )(x)
        
        x = nn.Dense(self.lst_layer[-1])(x).squeeze()
        return x



# %%
def modelPP_NN(hyperparams, X, Y=None, is_training=False):
    
    lst_layer = hyperparams['lst_layer']
    lst_dropout = hyperparams['lst_dropout']
    assert len(lst_layer) == len(lst_dropout) + 1

    nn_logits = random_flax_module(
                "mlp",
                MLP(lst_layer, lst_dropout),
                input_shape=(1, X.shape[1]),
                prior={
                    **{f"Dense_{i}.bias":dist.Cauchy()   for i in range(0,len(lst_layer))},
                    **{f"Dense_{i}.kernel":dist.Normal()  for i in range(0,len(lst_layer))}
                    },
                # ensure PRNGKey is made available to dropout layers
                apply_rng=["dropout"],
                # indicate mutable state due to BatchNorm layers
                # mutable=["batch_stats"],
                # to ensure proper initialisation of BatchNorm we must
                # initialise with is_training=True
                is_training=True,
            )
    rng_key = hyperparams['rng_key']
    with numpyro.plate("samples", X.shape[0], subsample_size=hyperparams["batch_size"]):
        batch_X = numpyro.subsample(X, event_dim=1)
        
        batch_Y = numpyro.subsample(Y, event_dim=0) if Y is not None else None

        rng_key, _rng_key = jax.random.split(key=rng_key)
        logits = nn_logits(batch_X, is_training, rngs={"dropout": _rng_key})
        
        numpyro.deterministic("logits", logits)

        numpyro.sample("Y", dist.BernoulliLogits(logits=logits), obs=batch_Y)
        
        
# %%
if __name__ == '__main__':
    """This example shows how the NN capabilities can be used for a "degenerated" NN
    without any hidden layer and no dropout. This is basically a logistic regression.
    This is not really useful, but might be useful to understand what is going on.
    
    """
    
    hyperparams = {}
    hyperparams['N'] = 100000
    hyperparams['K'] = 5
    hyperparams['rng_key'] = rng_key
    hyperparams['batch_size'] = hyperparams['N']
    # Specify a degenerated NN with one layer and no dropout, similar to a logistic regression
    hyperparams['lst_layer'] = [1]
    hyperparams['lst_dropout'] = []

    X = np.random.normal(0, 1.0, size=(hyperparams['N'],hyperparams['K']))

    # Run the DGP  once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(modelPP_NN, num_samples=1)
    lat_samples = lat_predictive(rng_key_,X=X,hyperparams=hyperparams)

    coefTrue = {s:lat_samples[s][0] for s in lat_samples.keys() if s not in ['Y','logits']}
    print(coefTrue)
    # %%
    # # Condition the model and get predictions for Y
    condition_model = numpyro.handlers.condition(modelPP_NN, data=coefTrue)
    nPriorSamples = 1
    prior_predictive = Predictive(condition_model, num_samples=nPriorSamples, return_sites=["Y"])
    prior_samples = prior_predictive(rng_key_,X=X,hyperparams=hyperparams)
    Y = prior_samples['Y'].squeeze()
    Y.shape

    # %%
    # Estimate with SVI
    rng_key, rng_key_ = random.split(rng_key)
    guide = autoguide.AutoNormal(modelPP_NN, 
                        init_loc_fn=init_to_feasible)


    hyperparams['batch_size'] = 100

    svi = SVI(modelPP_NN,guide,optim.Adam(0.1),Trace_ELBO())
    svi_result = svi.run(rng_key_, 1000,X=X,Y=Y,hyperparams=hyperparams,is_training=True)
    svi_params = svi_result.params
    # %%
    # Get samples from the posterior
    predictive = Predictive(guide, params=svi_params, num_samples=500)
    samples_svi = predictive(random.PRNGKey(1), X=X,hyperparams=hyperparams,is_training=False)
    samples_svi.keys()
    # %%
    samples_svi['mlp/Dense_0.kernel'].mean(axis=0)
    # %%
    coefTrue['mlp/Dense_0.kernel']
    # %%
    plt.hist(samples_svi['mlp/Dense_0.kernel'][:,0,0],bins=50);
    plt.hist(samples_svi['mlp/Dense_0.bias'].squeeze(),bins=50);

    coefTrue['mlp/Dense_0.bias']
    # %%
    # Get posterior predictions using samples from the posterior
    hyperparams['batch_size'] = hyperparams['N']
    predictivePosterior = Predictive(modelPP_NN, posterior_samples=samples_svi)
    post_predict = predictivePosterior(random.PRNGKey(1), X=X, hyperparams=hyperparams,is_training=False)
    post_predict.keys()

    # %%
    # Manually check that the logits are correct for the first sample
    aa = X @ samples_svi['mlp/Dense_0.kernel'][0,] + samples_svi['mlp/Dense_0.bias'][0].squeeze()
    aa.shape
    assert np.allclose(aa[:,0],post_predict['logits'][0,:],atol=1e-5)



# %%
