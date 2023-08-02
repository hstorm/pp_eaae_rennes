# %%
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
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
def modelPotOutcome(X, T=None, Y=None):
    
    alpha_out = numpyro.sample("alpha_out", dist.Normal(0.,1).expand([X.shape[1]]))
    beta_treat = numpyro.sample("beta_treat", dist.Normal(0.,1).expand([X.shape[1]]))
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))

    Y0 = X @ alpha_out 
    tau = X @ beta_treat 
    Y1 = Y0 + tau 
    logits_T = Y1 - Y0 
    T = numpyro.sample("T", dist.Bernoulli(logits=logits_T), obs=T)
    numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=Y)
    
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("Y1", Y1)
        


# %%
def modelLinearEffects():
    # %%
    N = 10000
    K = 5
    X = np.random.normal(0, 1.0, size=(N,K))

    # Run the DGP  once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(modelPP_treament, num_samples=1)
    lat_samples = lat_predictive(rng_key_,X=X)
    lat_samples['Y0'].shape

    # %%
    # # Condition the model and get predictions for Y
    condition_model = numpyro.handlers.condition(modelPP_treament, data=coefTrue)
    nPriorSamples = 1
    prior_predictive = Predictive(condition_model, num_samples=nPriorSamples)
    prior_samples = prior_predictive(rng_key_,X=X)
    Y = prior_samples['Y'].squeeze()
    T = prior_samples['T'].squeeze()
    mu_Y0 = prior_samples['Y0'].squeeze()
    mu_Y1 = prior_samples['Y1'].squeeze()
    beta_true = prior_samples['beta_treat'].squeeze()
    alpha_true = prior_samples['alpha_out'].squeeze()
    print('avg treatment effect',np.mean(mu_Y1-mu_Y0))
    plt.hist(mu_Y1-mu_Y0,bins=100);
    
    # %%
    # Estimate with SVI
    rng_key, rng_key_ = random.split(rng_key)
    guide = autoguide.AutoNormal(modelPP_treament, 
                        init_loc_fn=init_to_feasible)

    # svi = SVI(modelPP_treament,guide,optim.Adam(0.01),Trace_ELBO())
    svi = SVI(modelPP_treament,guide,optim.Adam(0.01),TraceMeanField_ELBO())
    svi_result = svi.run(rng_key_, 10000,X=X,T=T,Y=Y)
    plt.plot(svi_result.losses)
    svi_params = svi_result.params
    # %%
    # Get samples from the posterior
    predictive = Predictive(guide, params=svi_params, num_samples=500)
    samples_svi = predictive(random.PRNGKey(1), X=X)
    samples_svi.keys()
    # %%
    # Get posterior predictions using samples from the posterior
    hyperparams['batch_size'] = hyperparams['N']
    predictivePosterior = Predictive(modelPP_treament, posterior_samples=samples_svi)
    post_predict = predictivePosterior(random.PRNGKey(1), X=X)
    post_predict.keys()
    
    print('true avg treatment effect',np.mean(mu_Y1-mu_Y0))
    print('estimated avg treatment effect',np.mean(post_predict['mu_Y1']-post_predict['mu_Y0']))
    
    print('alpha_true',alpha_true)
    print('alpha_hat',np.mean(samples_svi['alpha_out'],axis=0))
    
    print('beta_true',beta_true)
    print('beta_hat',np.mean(samples_svi['beta_treat'],axis=0))

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
    use_bias: Sequence[float]

    @nn.compact
    def __call__(self, x, is_training:bool):
        assert len(self.lst_layer) == len(self.dropout_rates) + 1
        assert len(self.lst_layer) == len(self.use_bias) + 1
        
        for iLayer in range(0,len(self.lst_layer[:-1])):
            # x = nn.tanh(nn.Dense(self.lst_layer[iLayer])(x))
            # x = nn.relu(nn.Dense(self.lst_layer[iLayer])(x))
            x = nn.leaky_relu(nn.Dense(self.lst_layer[iLayer],use_bias=self.use_bias[iLayer])(x))
        
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
def modelPP_NN_treament(hyperparams, X, T=None, Y=None, is_training=False):
    
    lst_layer_potOut = hyperparams['lst_layer_potOut']
    lst_dropout_potOut = hyperparams['lst_dropout_potOut']
    lst_use_bias_potOut = hyperparams['lst_use_bias_potOut']
    
    lst_layer_treatEffect = hyperparams['lst_layer_treatEffect']
    lst_dropout_treatEffect = hyperparams['lst_dropout_treatEffect']
    lst_use_bias_treatEffect = hyperparams['lst_use_bias_treatEffect']
    
    assert len(lst_layer_potOut) == len(lst_dropout_potOut) + 1
    assert len(lst_layer_potOut) == len(lst_use_bias_potOut) + 1
    assert len(lst_layer_treatEffect) == len(lst_dropout_treatEffect) + 1
    assert len(lst_layer_treatEffect) == len(lst_use_bias_treatEffect) + 1

    # Specify a NN for the potential outcomes without the treatment effect
    nn_potOut = random_flax_module("nn_potOut",
                MLP(lst_layer_potOut, lst_dropout_potOut,lst_use_bias_potOut),
                input_shape=(1, X.shape[1]), 
                prior={**{f"Dense_{i}.bias":dist.Cauchy()   for i in range(0,len(lst_layer_potOut))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1)  for i in range(0,len(lst_layer_potOut))}},
                apply_rng=["dropout"],is_training=True)
    
    # Specify the treatment effect parameter as a NN
    nn_treatEffect = random_flax_module("nn_treatEffect",
                MLP(lst_layer_treatEffect, lst_dropout_treatEffect,lst_use_bias_treatEffect),
                input_shape=(1, X.shape[1]), 
                prior={**{f"Dense_{i}.bias":dist.Cauchy()   for i in range(0,len(lst_layer_treatEffect))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1)  for i in range(0,len(lst_layer_treatEffect))}},
                apply_rng=["dropout"],is_training=True)

    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))
    
    rng_key = hyperparams['rng_key']
    with numpyro.plate("samples", X.shape[0], subsample_size=hyperparams["batch_size"]):
        batch_X = numpyro.subsample(X, event_dim=1)
        batch_Y = numpyro.subsample(Y, event_dim=0) if Y is not None else None
        batch_T = numpyro.subsample(T, event_dim=0) if T is not None else None
        # Calculate expect potOut for T=0
        rng_key, _rng_key = jax.random.split(key=rng_key)
        mu_P0 = nn_potOut(batch_X, is_training, rngs={"dropout": rng_key})
        # Calculate individual specific treatment effect
        rng_key, _rng_key = jax.random.split(key=rng_key)
        b_treat = nn_treatEffect(batch_X, is_training, rngs={"dropout": rng_key})
        # Calculate expect potOut for T=1 as mu_P0 plus individual specific treatment effect
        mu_P1 = mu_P0 + b_treat
        # Logits for T are the difference between the two expected potential outcomes
        logits_T = mu_P1 - mu_P0
        T = numpyro.sample("T", dist.Bernoulli(logits=logits_T), obs=batch_T)
        # Observed outcome depends on mu_P1 and mu_P0 depending on T
        numpyro.sample("Y", dist.Normal(mu_P1*T + mu_P0*(1-T), sigma_Y), obs=batch_Y)
        
        numpyro.deterministic("b_treat", b_treat)
        numpyro.deterministic("mu_P0", mu_P0)
        numpyro.deterministic("mu_P1", mu_P1)
        

    
# %%
if __name__ == '__main__':
  """Check if training work for a "deep" net with 2 hidden layers 
    """
    # %%
    hyperparams = {}
    hyperparams['N'] = 10000
    hyperparams['K'] = 5
    hyperparams['rng_key'] = rng_key
    hyperparams['batch_size'] = hyperparams['N']
    # Specify a degenerated NN with one layer and no dropout, similar to a logistic regression
    hyperparams['lst_layer_potOut'] = [512,64,1]
    hyperparams['lst_dropout_potOut'] = [0.0,0.0]
    hyperparams['lst_use_bias_potOut'] = [False,False]
    hyperparams['lst_layer_treatEffect'] = [1]
    hyperparams['lst_dropout_treatEffect'] = []
    hyperparams['lst_use_bias_treatEffect'] = []

    X = np.random.normal(0, 1.0, size=(hyperparams['N'],hyperparams['K']))

    # Run the DGP  once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(modelPP_NN_treament, num_samples=1)
    lat_samples = lat_predictive(rng_key_,X=X,hyperparams=hyperparams)

    coefTrue = {s:lat_samples[s][0] for s in 
                lat_samples.keys() if s not in ['Y','T','mu_P0', 'mu_P1','b_treat']}
    coefTrue.keys()
    # %%
    # # Condition the model and get predictions for Y
    condition_model = numpyro.handlers.condition(modelPP_NN_treament, data=coefTrue)
    nPriorSamples = 1
    prior_predictive = Predictive(condition_model, num_samples=nPriorSamples, 
                                  return_sites=["Y",'T','mu_P0', 'mu_P1','b_treat'])
    prior_samples = prior_predictive(rng_key_,X=X,hyperparams=hyperparams)
    Y = prior_samples['Y'].squeeze()
    T = prior_samples['T'].squeeze()
    mu_P0 = prior_samples['mu_P0'].squeeze()
    mu_P1 = prior_samples['mu_P1'].squeeze()
    b_treat = prior_samples['b_treat'].squeeze()
    print('Min,Max,Mean Y',np.min(Y),np.max(Y),np.mean(Y))
    print('Min,Max,Mean T',np.min(T),np.max(T),np.mean(T))
    plt.hist(Y[:10000],bins=100);
    print('Min,Max,Mean mu_P0',np.min(mu_P0),np.max(mu_P0),np.mean(mu_P0))
    print('Min,Max,Mean mu_P1',np.min(mu_P1),np.max(mu_P1),np.mean(mu_P1))
    plt.show()
    print('avg treatment effect',np.mean(mu_P1-mu_P0))
    plt.hist(mu_P1-mu_P0,bins=100);
    
    # %%
    mu_diff = mu_P1-mu_P0
    aa = pd.DataFrame(np.hstack([X,mu_diff[:,None]]),
                      columns=[f'X{i}' for i in range(0,X.shape[1])]+['mu_diff'])
    x_vars = [f'X{i}' for i in range(0,X.shape[1])]
    y_vars = ["mu_diff"]
    
    g = sns.PairGrid(aa,x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot,s=0.1)
    # %%  
    Xc = np.hstack([np.ones([T.shape[0],1]), T[:,None],X])
    b_hat_ols = np.linalg.inv(Xc.T@Xc)@Xc.T@Y
    y_hat_ols = Xc @ b_hat_ols
    
    plt.scatter(Y,y_hat_ols,s=0.1)
    # calculate R2
    import sklearn.metrics as metrics
    r2_ols = metrics.r2_score(Y, y_hat_ols)
    print('R2 OLS',r2_ols)
    b_treatment_hat = b_hat_ols[1]
    print('b_treatment_hat',b_treatment_hat)
    print('avg treatment effect',np.mean(mu_P1-mu_P0))
    # %%
    # numpyro.render_model(modelPP_NN_treament, model_args=(hyperparams,X),
    #                      render_distributions=True)
    # %%
    # Estimate with SVI
    rng_key, rng_key_ = random.split(rng_key)
    guide = autoguide.AutoNormal(modelPP_NN_treament, 
                        init_loc_fn=init_to_feasible)

    hyperparams['lst_layer_potOut'] = [256,32,1]
    hyperparams['lst_dropout_potOut'] = [0.2,0.1]
    hyperparams['lst_use_bias_potOut'] = [True,True]
    hyperparams['lst_layer_treatEffect'] = [32,1]
    hyperparams['lst_dropout_treatEffect'] = [0.1]
    hyperparams['lst_use_bias_treatEffect'] = [True]
    hyperparams['batch_size'] = 512

    # svi = SVI(modelPP_NN_treament,guide,optim.Adam(0.01),Trace_ELBO())
    svi = SVI(modelPP_NN_treament,guide,optim.Adam(0.01),TraceMeanField_ELBO())
    svi_result = svi.run(rng_key_, 10000,X=X,T=T,Y=Y,hyperparams=hyperparams,is_training=True)
    plt.plot(svi_result.losses)
    svi_params = svi_result.params
    # %%
    # Get samples from the posterior
    predictive = Predictive(guide, params=svi_params, num_samples=500)
    samples_svi = predictive(random.PRNGKey(1), X=X,hyperparams=hyperparams,is_training=False)
    samples_svi.keys()
    # %%
    # Get posterior predictions using samples from the posterior
    hyperparams['batch_size'] = hyperparams['N']
    predictivePosterior = Predictive(modelPP_NN_treament, posterior_samples=samples_svi, 
                                     return_sites=['Y','mu_P0', 'mu_P1','b_treat'])
    post_predict = predictivePosterior(random.PRNGKey(1), X=X, hyperparams=hyperparams,is_training=False)
    post_predict.keys()

    # %%
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(logits_true,post_predict['logits'].mean(axis=0),s=0.1)
    # ax.set_ylim([logits_true.min(),logits_true.max()]);
    # ax.set_xlim([logits_true.min(),logits_true.max()]);
    ax.scatter(Y,post_predict['Y'].mean(axis=0),s=0.1)
    # ax.set_ylim([Y.min(),Y.max()]);
    # ax.set_xlim([Y.min(),Y.max()]);

    r2_svi = metrics.r2_score(Y, post_predict['Y'].mean(axis=0))
    print('R2 SVI',r2_svi)
    # %%
    plt.scatter(b_treat,post_predict['b_treat'].mean(axis=0),s=0.1)
    # %%
    plt.scatter(mu_P0,post_predict['mu_P0'].mean(axis=0),s=0.1)
    plt.scatter(mu_P1,post_predict['mu_P1'].mean(axis=0),s=0.1)
    print('true avg treatment effect',np.mean(mu_P1-mu_P0))
    print('estimated avg treatment effect',np.mean(post_predict['mu_P1']-post_predict['mu_P0']))
    # %%
    mu_diff_predict = np.mean(post_predict['mu_P1']-post_predict['mu_P0'],axis=0)
    aa = pd.DataFrame(np.hstack([X,mu_diff_predict[:,None]]),
                      columns=[f'X{i}' for i in range(0,X.shape[1])]+['mu_diff_predict'])
    x_vars = [f'X{i}' for i in range(0,X.shape[1])]
    y_vars = ["mu_diff_predict"]
    
    g = sns.PairGrid(aa,x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot,s=0.1)
    for i in range(0,X.shape[1]):
        g.axes[0,1].set_ylim(-10,10)
# %%
