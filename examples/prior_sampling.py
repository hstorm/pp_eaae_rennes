# %%
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

import jax.numpy as jnp
from jax import nn, lax, random
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive

# SVI 
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoDiagonalNormal,AutoMultivariateNormal
import numpyro.optim as optim

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
# numpyro.set_platform("gpu")
# numpyro.set_host_device_count(2)

# Adjust to own setting (correct for VS code devcontainer)
os.chdir("/workspaces/pp_eaae_rennes/")

# %%
rng_key = random.PRNGKey(1)

# %%
# Load data
from util.load_yield_data import getData
dfL_train, dfL_test, lstCatCrop, lstCatNUTS3, lstSmi25, lstSmi180, scale_train = getData()   

lstColX = ['bodenzahl_scaled',
           'OCT_25',
            'NOV_25',
            'DEZ_25',
            'JAN_25',
            'FEB_25',
            'MAR_25',
            'APR_25',
            'MAI_25',
            'JUN_25',
            'JUL_25',
            'AUG_25'
            ] 


dfWheat_train = dfL_train.loc[dfL_train['crop']=='Winterweizen',:]
    
X = dfL_train.loc[dfL_train['crop']=='Winterweizen',lstColX].values 
Y = dfL_train.loc[dfL_train['crop']=='Winterweizen','yield_scaled'].values    


# %%
# =============================================================================
# Define most basic linear regression model
# =============================================================================
def model(X,sigma_b, Y=None):
    b = numpyro.sample('b', dist.Normal(0,sigma_b).expand([X.shape[1]]))
    # b = numpyro.sample('b', dist.Uniform(0,1).expand([X.shape[1]]))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('Y',dist.Normal(X @ b,sigma), obs=Y)

# =============================================================================
# Prior sampling
# =============================================================================
sigma_b = 3
nPriorSamples = 10000
# Prior sampling
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(model, num_samples=nPriorSamples)
prior_samples = prior_predictive(rng_key_,X=X,sigma_b=sigma_b)
# 
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist((prior_samples['Y'][:,0]*scale_train['Winterweizen_yield_std']+scale_train['Winterweizen_yield_mean'])/10,bins=100,
        density=True,
        color='grey');
ax.set_title(f'b~N(0,{sigma_b})', fontsize=20)
ax.set_xlabel('Yield [t/ha]', fontsize=20)
ax.set_ylabel('Density', fontsize=20)
# ax.get_yaxis().set_visible(False)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
# %%
x_range_scaled = np.linspace(-5,5,100)
# x_mean = X.mean(axis=0)*scale_train[lstColX].std(axis=0)+scale_train[lstColX].mean(axis=0)
x_mean_scaled = X.mean(axis=0)
x_plot = np.repeat(x_mean_scaled.reshape(1,-1),100,axis=0)
x_plot[:,0] = x_range_scaled
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
x_range = x_range_scaled*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean']
for i in range(1,300):
    y_hat_scaled = x_plot @ prior_samples['b'][i,:].reshape(-1,1) 
    
    y_hat = y_hat_scaled*scale_train['Winterweizen_yield_std']+scale_train['Winterweizen_yield_mean']

    ax.plot(x_range,y_hat/10,color='k',alpha=0.2)

ax.set_title(f'b~N(0,{sigma_b})', fontsize=20)    
ax.set_xlabel('Soil Rating [0-100]', fontsize=20)
ax.set_ylabel('Yield [t/ha]', fontsize=20)
ax.set_xlim([0,100])
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

# %%
print(f"SoilRating [0-100]: Mean={scale_train['bodenzahl_mean']:.2f}, Std={scale_train['bodenzahl_std']:.2f}")
print(f"WinterWheatYield: Mean={scale_train['Winterweizen_yield_mean']:.2f}dt, Std={scale_train['Winterweizen_yield_std']:.2f}dt")

# %%
# Estimate model using numpyro MCMC
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=2)
mcmc.run(rng_key_, X=X, sigma_b=sigma_b, Y=Y)
mcmc.print_summary()

azMCMC = az.from_numpyro(mcmc)
azMCMC= azMCMC.assign_coords({
                'b_dim_0':lstColX,
                    })
az.summary(azMCMC)
# %%
# Get posterior samples
post_samples = mcmc.get_samples()
# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(prior_samples['b'][:,0],bins=100,density=True, label='prior', color='grey');
ax.hist(post_samples['b'][:,0],bins=100,density=True, label='posterior', color='black');
ax.set_title(f'b~N(0,{sigma_b})', fontsize=20)
# ax.set_xlabel(f"b[{lstColX[0]}]", fontsize=20)
ax.set_xlabel(f"b[{'Soil Rating'}]", fontsize=20)
ax.set_xlim([-3,3])
# ax.get_yaxis().set_visible(False)
ax.set_ylabel('Density', fontsize=20)
ax.legend()
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
# %%
x_range_scaled = np.linspace(-5,5,100)
# x_mean = X.mean(axis=0)*scale_train[lstColX].std(axis=0)+scale_train[lstColX].mean(axis=0)
x_mean_scaled = X.mean(axis=0)
x_plot = np.repeat(x_mean_scaled.reshape(1,-1),100,axis=0)
x_plot[:,0] = x_range_scaled
fig, ax = plt.subplots(1, 1, figsize=(8, 4))


x_range = x_range_scaled*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean']
for i in range(1,300):
    y_hat_scaled = x_plot @ post_samples['b'][i,:].reshape(-1,1) 
    
    y_hat = y_hat_scaled*scale_train['Winterweizen_yield_std']+scale_train['Winterweizen_yield_mean']


    ax.plot(x_range,y_hat/10,color='k',alpha=0.2)

ax.set_title(f'b~N(0,{sigma_b})', fontsize=20)    
ax.set_xlabel('Soil Rating [0-100]', fontsize=20)
ax.set_ylabel('Yield [t/ha]', fontsize=20)
ax.set_xlim([0,100])
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(X[:,0]*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean'],density=True, color='grey');
ax.set_xlabel("Soil rating")
ax.set_ylabel("Density")
ax.set_xlim([0,100])

# %%
