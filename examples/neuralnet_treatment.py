# %%
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

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
# numpyro.set_platform("gpu")
# numpyro.set_host_device_count(2)

# %%
rng_key = random.PRNGKey(1)

# %%
def modelPotOutcome(X, T=None, Y=None):
    """Define a simple potential outcome model

    Args:
        X (np array): Explanatory variables (standardized)
        T (np array, optional): Treatment status. Defaults to None.
        Y (np array, optional): Observed outcome. Defaults to None.
    """
    alpha_out = numpyro.sample("alpha_out", dist.Normal(0.,1).expand([X.shape[1]]))
    beta_treat = numpyro.sample("beta_treat", dist.Normal(0.,1).expand([X.shape[1]]))
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))

    Y0 = X @ alpha_out 
    tau = X @ beta_treat 
    Y1 = Y0 + tau 
    T = numpyro.sample("T", dist.Bernoulli(logits=Y1 - Y0), obs=T)
    numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=Y)
    
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("Y1", Y1)
        
# %%
def modelPotOutcome_poly(X, polyDegree=1, stepFunction=False, T=None, Y=None):
    """Extented potential outcome model with polynomial terms and 
    step function for the treatment effect

    Args:
        X (_type_): Explanatory variables (standardized)
        polyDegree (int, optional): Degree of polynomial. Defaults to 1.
        stepFunction (bool, optional): Indicator to use a step function for 
            first explanatory variable. Defaults to False.
        T (np array, optional): Treatment status. Defaults to None.
        Y (np array, optional): Observed outcome. Defaults to None.
    """
    alpha_out = numpyro.sample("alpha_out", dist.Normal(0.,1).expand([X.shape[1]]))
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))
    
    beta_treat = numpyro.sample("beta_treat", dist.Normal(0.,1).expand([X.shape[1]]))
    tau = X @ beta_treat 
    if polyDegree>1:
        betaSq_treat = numpyro.sample("betaSq_treat", dist.Normal(0.,1).expand([X.shape[1]]))
        tau = tau + X**2 @ betaSq_treat 

        if polyDegree>2:
            betaCub_treat = numpyro.sample("betaCub_treat", dist.Normal(0.,1).expand([X.shape[1]]))
            tau = tau + X**3 @ betaCub_treat 

    if stepFunction:
        betaStep_treat = numpyro.sample("betaStep_treat", dist.Normal(0.,5)) 
        tau = tau + betaStep_treat * (X[:,0]>0.0)

    Y0 = X @ alpha_out 
    Y1 = Y0 + tau 
    T = numpyro.sample("T", dist.Bernoulli(logits=Y1 - Y0), obs=T)
    numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=Y)
    
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("Y1", Y1)

# %%
# Define a Flax NN model class which can be used within the DGP
# Useful source: https://omarfsosa.github.io/bayesian_nn
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
            x = nn.leaky_relu(nn.Dense(self.lst_layer[iLayer],use_bias=self.use_bias[iLayer])(x))
        
            if self.dropout_rates[iLayer] > 0.0:
                x = nn.Dropout(self.dropout_rates[iLayer], 
                    deterministic=not is_training)(x)
        
        # FIXME batch norm not implemented and not yet easily working with numpyro
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
    """Potential outcome model using neural networks (dense MPL) for 
    the treatment effect and the not treated outcome

    Args:
        hyperparams (dict): dict to specify number of layers, dropout, batchsize,
        X (_type_): Explanatory variables (standardized)
        T (np array, optional): Treatment status. Defaults to None.
        Y (np array, optional): Observed outcome. Defaults to None.
        is_training (bool, optional): Set to true for inference. Defaults to False.
    """
    lst_lay_Y0 = hyperparams['lst_lay_Y0']
    lst_drop_Y0 = hyperparams['lst_drop_Y0']
    lst_bias_Y0 = hyperparams['lst_bias_Y0']
    
    lst_lay_tau = hyperparams['lst_lay_tau']
    lst_drop_tau = hyperparams['lst_drop_tau']
    lst_bias_tau = hyperparams['lst_bias_tau']
    
    assert len(lst_lay_Y0) == len(lst_drop_Y0) + 1
    assert len(lst_lay_Y0) == len(lst_bias_Y0) + 1
    assert len(lst_lay_tau) == len(lst_drop_tau) + 1
    assert len(lst_lay_tau) == len(lst_bias_tau) + 1

    # Specify a NN for the potential outcomes without the treatment effect
    prior_MLP_Y0 = {**{f"Dense_{i}.bias":dist.Cauchy() for i in range(0,len(lst_lay_Y0))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1) for i in range(0,len(lst_lay_Y0))}}
    MLP_Y0 = random_flax_module("MLP_Y0",
                MLP(lst_lay_Y0, lst_drop_Y0,lst_bias_Y0),
                input_shape=(1, X.shape[1]), 
                prior=prior_MLP_Y0,
                apply_rng=["dropout"],is_training=True)

    # Specify the treatment effect parameter as a NN
    prior_MPL_tau = {**{f"Dense_{i}.bias":dist.Cauchy() for i in range(0,len(lst_lay_tau))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1) for i in range(0,len(lst_lay_tau))}}
    MLP_tau = random_flax_module("MLP_tau",
                MLP(lst_lay_tau, lst_drop_tau,lst_bias_tau),
                input_shape=(1, X.shape[1]), 
                prior=prior_MPL_tau,
                apply_rng=["dropout"],is_training=True)

    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))
    
    rng_key = hyperparams['rng_key']
    with numpyro.plate("samples", X.shape[0], subsample_size=hyperparams["batch_size"]):
        batch_X = numpyro.subsample(X, event_dim=1)
        batch_Y = numpyro.subsample(Y, event_dim=0) if Y is not None else None
        batch_T = numpyro.subsample(T, event_dim=0) if T is not None else None

        rng_key, _rng_key = jax.random.split(key=rng_key)
        Y0 = MLP_Y0(batch_X, is_training, rngs={"dropout": _rng_key})

        rng_key, _rng_key = jax.random.split(key=rng_key)
        tau = MLP_tau(batch_X, is_training, rngs={"dropout": _rng_key})

        Y1 = Y0 + tau
        T = numpyro.sample("T", dist.Bernoulli(logits=Y1 - Y0), obs=batch_T)
        numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=batch_Y)
        
        numpyro.deterministic("tau", tau)
        numpyro.deterministic("Y0", Y0)
        numpyro.deterministic("Y1", Y1)


# %%        
def data_generating(rng_key=rng_key,
                    modelTypeDataGen = 'linear',
                    N = 10000,
                    K = 5,
                    X=None): 
    """Helper function to generate data using different model types.
    Creates some basic plots to illustrate DGP

    Args:
        rng_key (_type_, optional): numpyro rng key. Defaults to rng_key.
        modelTypeDataGen (str, optional): Name of model type. Defaults to 'linear'.
        N (int, optional): number of samples. Defaults to 10000.
        K (int, optional): number of explanatory variables (ignored if X is provided). Defaults to 5.
        X (_type_, optional): Predefined exlanatory variables. Defaults to None.

    Raises:
        ValueError: if modelTypeDataGen is not recognized

    """
    # Generate X if not provided    
    if X is None:
        X = np.random.normal(0, 1.0, size=(N,K))

    if modelTypeDataGen == 'linear':
        model = modelPotOutcome
        datX_conditioned = {'X':X}
    elif modelTypeDataGen == 'poly2':
        model = modelPotOutcome_poly
        datX_conditioned = {'X':X, 'polyDegree':2}
    elif modelTypeDataGen == 'poly3':
        model = modelPotOutcome_poly
        datX_conditioned = {'X':X, 'polyDegree':3}
    elif modelTypeDataGen == 'poly3_step':
        model = modelPotOutcome_poly
        datX_conditioned = {'X':X,'stepFunction':True, 'polyDegree':3}
    elif modelTypeDataGen == 'NN':
        model = modelPP_NN_treament
        
        hyperparams = {}
        hyperparams['N'] = 10000
        hyperparams['K'] = 5
        hyperparams['rng_key'] = rng_key
        hyperparams['batch_size'] = hyperparams['N']
        hyperparams['lst_lay_Y0'] = [512,64,1]
        hyperparams['lst_drop_Y0'] = [0.0,0.0]
        hyperparams['lst_bias_Y0'] = [True,True]
        hyperparams['lst_lay_tau'] = [512,64,32,1]
        hyperparams['lst_drop_tau'] = [0.0,0.0,0.0]
        hyperparams['lst_bias_tau'] = [True,True,True]
        
        datX_conditioned = {'X':X, 'hyperparams':hyperparams}
    else:
        raise ValueError('modelTypeDataGen not recognized')

    # Run the DGP  once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(model, num_samples=1)
    lat_samples = lat_predictive(rng_key_,**datX_conditioned)
    lat_samples['Y0'].shape

    coefTrue = {s:lat_samples[s][0] for s in 
                lat_samples.keys() if s not in ['Y','T','Y0', 'Y1','b_treat']}
    coefTrue.keys()
    
    # Condition the model and get predictions for Y
    condition_model = numpyro.handlers.condition(model, data=coefTrue)
    conditioned_predictive = Predictive(condition_model, num_samples=1)
    prior_samples = conditioned_predictive(rng_key_,**datX_conditioned)
    Y_unscaled = prior_samples['Y'].squeeze()
    T = prior_samples['T'].squeeze()
    Y0 = prior_samples['Y0'].squeeze()
    Y1 = prior_samples['Y1'].squeeze()
    print('avg treatment effect',np.mean(Y1-Y0))
    plt.hist(Y1-Y0,bins=100);
        
    # Standardize Y
    Y_mean = Y_unscaled.mean(axis=0)
    Y_std = Y_unscaled.std(axis=0)
    Y = (Y_unscaled - Y_mean)/Y_std
    
    print('Share treated',np.mean(T))
    print(f'Mean(Y)={np.mean(Y):.4f}; std(Y)={np.std(Y):.4f}')
    
    # %
    if modelTypeDataGen != 'NN':
        beta_true = prior_samples['beta_treat'].squeeze()
        alpha_true = prior_samples['alpha_out'].squeeze()
    else:
        beta_true = {key:val for key, val in prior_samples.items() if 'MLP_tau' in key}
        alpha_true = {key:val for key, val in prior_samples.items() if 'MLP_Y0' in key}

    # Plot true Treatment heterogneity, for first covariate
    k = 0
    x_percentile = np.percentile(X[:,k],q=[2.5,97.5])
    x_range = np.linspace(x_percentile[0],x_percentile[1],100)
    x_mean = X.mean(axis=0)
    x_plot = np.repeat(x_mean.reshape(1,-1),100,axis=0)
    x_plot[:,k] = x_range
    
    datX_plot = datX_conditioned.copy()
    datX_plot['X'] = x_plot
    
    if modelTypeDataGen == 'NN':
        datX_plot['hyperparams']['batch_size'] = 100
    
    # Get prediction from the "true" conditioned model
    true_predict = conditioned_predictive(rng_key_,**datX_plot)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Plot "true" effect in red    
    ax.plot(x_plot[:,k],(true_predict['Y1']-true_predict['Y0'])[0,:],color='r',alpha=1);

    ax.set_xlabel(f'X[{k}]', fontsize=20)
    ax.set_ylabel('tau', fontsize=20)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    # Plot hist of Y0 and Y1
    fig, ax = plt.subplots()
    ax.hist(Y[T==0][:10000],bins=100,density=True,color='green',alpha=0.5,label='T=0');
    ax.hist(Y[T==1][:10000],bins=100,density=True,color='red',alpha=0.5,label='T=1');
    
    # Plot scatter of tau vs X    
    mu_diff = Y1-Y0
    aa = pd.DataFrame(np.hstack([X,mu_diff[:,None]]),
                      columns=[f'X{i}' for i in range(0,X.shape[1])]+['tau'])
    x_vars = [f'X{i}' for i in range(0,X.shape[1])]
    y_vars = ["tau"]
    
    g = sns.PairGrid(aa,x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot,s=0.1)
    
    return Y, Y_unscaled, Y_mean, Y_std, T, X, Y0, Y1, beta_true, alpha_true, conditioned_predictive, datX_conditioned

# %%
if __name__ == '__main__':

    # %%
    # =====================================================
    # Generate the data, set modelType to desired name
    # =====================================================
    N = 200000
    K = 5
    # modelTypeDataGen = 'linear'
    # modelTypeDataGen = 'poly2'
    # modelTypeDataGen = 'poly3'
    modelTypeDataGen = 'poly3_step'
    # modelTypeDataGen = 'NN'
    # use a see that happen to produce a good split between treated and untreated for the NN
    # rng_key = jnp.array([3599756002, 4216389472], dtype='uint32') 
    rng_key = jnp.array([0, 1], dtype='uint32') 
    # print(rng_key)
    # [2441914641 1384938218]
    rng_key, rng_key_ = random.split(rng_key)
    Y, Y_unscaled, Y_mean, Y_std, T, X, Y0_true, Y1_true, beta_true, alpha_true, conditioned_predictive, datX_conditioned = data_generating(
        rng_key=rng_key,
        modelTypeDataGen = modelTypeDataGen,
        N = N,
        K = 5) 
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(X[T==0,0],bins=100, label='T=0');
    ax.hist(X[T==1,0],bins=100, label='T=1');
    ax.set_xlabel(f'X[0]', fontsize=20)
    ax.legend()

    # %%
    # =====================================================
    # Estimate model, set inference model to desired model
    # =====================================================
    # modelTypeInference = 'linear'
    # modelTypeInference = 'poly2'
    # modelTypeInference = 'poly3'
    modelTypeInference = 'NN'
    if modelTypeInference == 'linear':
        model = modelPotOutcome
        datXY = {'X':X, 'Y':Y_unscaled, 'T':T}
        datX = {'X':X, 'T':T}
    elif modelTypeInference == 'poly2':
        model = modelPotOutcome_poly
        datXY = {'X':X, 'Y':Y_unscaled, 'T':T, 'polyDegree':2}
        datX = {'X':X, 'T':T, 'polyDegree':2}
    elif modelTypeInference == 'poly3':
        model = modelPotOutcome_poly
        datXY = {'X':X, 'Y':Y_unscaled, 'T':T, 'polyDegree':3}
        datX = {'X':X, 'T':T, 'polyDegree':3}
    elif modelTypeInference == 'NN':
        model = modelPP_NN_treament
        
        hyperparams = {}
        hyperparams['N'] = N
        hyperparams['K'] = K
        hyperparams['rng_key'] = rng_key
        hyperparams['batch_size'] = 512
        hyperparams['lst_lay_Y0'] = [512,64,1]
        hyperparams['lst_drop_Y0'] = [0.2,0.2]
        hyperparams['lst_bias_Y0'] = [True,True]
        hyperparams['lst_lay_tau'] = [512,64,32,1]
        hyperparams['lst_drop_tau'] = [0.2,0.2,0.2]
        hyperparams['lst_bias_tau'] = [True,True,True]
        
        datXY = {'X':X, 'Y':Y_unscaled, 'T':T, 'hyperparams':hyperparams,'is_training':True}
        datX = {'X':X, 'T':T,  'hyperparams':hyperparams, 'is_training':False}
    else:
        raise ValueError('modelTypeInference not recognized')
    
    
    # Estimate with SVI
    rng_key, rng_key_ = random.split(rng_key)
    guide = autoguide.AutoNormal(model, 
                        init_loc_fn=init_to_feasible)

    svi = SVI(model,guide,optim.Adam(0.01),Trace_ELBO())
    svi_result = svi.run(rng_key_, 4000,**datXY)
    plt.plot(svi_result.losses)
    svi_params = svi_result.params
    
    # %%
    # Get samples from the posterior
    predictive = Predictive(guide, params=svi_params, num_samples=500)
    samples_svi = predictive(random.PRNGKey(1), **datX)
    samples_svi.keys()
    # %%
    # Get posterior predictions using samples from the posterior
    predictivePosterior = Predictive(model, posterior_samples=samples_svi)
    post_predict = predictivePosterior(random.PRNGKey(1), **datX)
    post_predict.keys()
    
    tau_mean_true = np.mean(Y1_true-Y0_true)
    print('True: avg treatment effect',tau_mean_true)
    tau_mean_hat_scaled = np.mean(post_predict['Y1']-post_predict['Y0'])
    tau_mean_hat = np.mean((post_predict['Y1']*Y_std+Y_mean)-(post_predict['Y0']*Y_std+Y_mean))
    # tau_mean_hat = tau_mean_hat_scaled*Y_std+Y_mean
    print('Estimated (scaled): avg treatment effect',tau_mean_hat_scaled)
    print('Estimated: avg treatment effect',tau_mean_hat)
    
    if modelTypeInference != 'NN':    
        print('alpha_true',alpha_true)
        print('alpha_hat',np.mean(samples_svi['alpha_out'],axis=0))
        
        print('beta_true',beta_true)
        print('beta_hat',np.mean(samples_svi['beta_treat'],axis=0))

    # %%
    k = 0
    x_percentile = np.percentile(X[:,k],q=[2.5,97.5])
    x_range = np.linspace(x_percentile[0],x_percentile[1],100)
    x_mean = X.mean(axis=0)
    x_plot = np.repeat(x_mean.reshape(1,-1),100,axis=0)
    x_plot[:,k] = x_range
    
    datX_plot = datX.copy()
    datX_plot['X'] = x_plot
    datX_plot['T'] = jnp.zeros(100)
    if modelTypeInference == 'NN':
        datX_plot['hyperparams']['batch_size'] = 100
    
    datX_plot_conditioned = datX_conditioned.copy()
    datX_plot_conditioned['X'] = x_plot
    datX_plot_conditioned['T'] = jnp.zeros(100)
    # Get posterior predictions
    post_predict = predictivePosterior(random.PRNGKey(1), **datX_plot)
    # Get prediction from the "true" conditioned model
    true_predict = conditioned_predictive(rng_key_,**datX_plot_conditioned)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(1,300):
        # tau_i = ((post_predict['Y1']*Y_std+Y_mean)-(post_predict['Y0']*Y_std+Y_mean))[i,:]
        tau_i = ((post_predict['Y1'])-(post_predict['Y0']))[i,:]
        # tau_i = tau_i_scaled*Y_std+Y_mean
        ax.plot(x_plot[:,k],tau_i,color='k',alpha=0.2);
    # Add "true" effect in red    
    ax.plot(x_plot[:,k],(true_predict['Y1']-true_predict['Y0'])[0,:],color='r',alpha=1);

    ax.set_xlabel(f'X[{k}]', fontsize=20)
    ax.set_ylabel('tau', fontsize=20)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
        
    
    # %%
    
        

    

