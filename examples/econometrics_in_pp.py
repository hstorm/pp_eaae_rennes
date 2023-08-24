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

# if "SVG" in os.environ:
#     %config InlineBackend.figure_formats = ["svg"]
az.style.use("arviz-darkgrid")
# numpyro.set_platform("cpu")
numpyro.set_platform("gpu")
# numpyro.set_host_device_count(2)

os.chdir("/home/storm/Research/pp_eaae_rennes")



# %%
rng_key = random.PRNGKey(1)

# %%
# =============================================================================
# Most basic linear regression model
# =============================================================================

# Generate artificial data
k = 4
N = 100
X = np.random.normal(size=[N, k]) 
b_true = np.arange(1,k+1)
sigma_true = 2
Y = np.dot(X, b_true) + np.random.normal(0,sigma_true,size=N)

# Define model
def model(X,Y=None):
    b = numpyro.sample('b', dist.Normal(0,1).expand([X.shape[1]]))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('Y',dist.Normal(X @ b,sigma), obs=Y)
# %%
# Estimate model using numpyro MCMC
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=500, num_warmup=1000, num_chains=2)
mcmc.run(rng_key_, X=X, Y=Y)
mcmc.print_summary()

# Compare to OLS: (X'X)^(-1)X'Y
b_hat_ols = np.linalg.inv(X.T@X)@X.T@Y
print('OLS:',b_hat_ols)
# %%
# Plot results using arviz
azMCMC = az.from_numpyro(mcmc)
az.plot_posterior(azMCMC, ref_val=list(b_true)+[sigma_true]);
az.plot_forest(azMCMC,combined=True, ridgeplot_overlap=1);
# Plots to inspect MCMC sampling
az.plot_trace(azMCMC, compact=True);
az.plot_rank(azMCMC);

az.plot_pair(azMCMC,
            kind='kde',
            divergences=True,
            textsize=18);
# %%
# Get posterior samples from MCMC
samples_mcmc = mcmc.get_samples()

# Perform Bayesian equivalent of a t-test 
# For example test how likely it is that b1 is smaller then 0.7,
# we calculate the share of posterior samples below 0.7
print('Prob below 0.7:', jnp.sum(samples_mcmc['b'][:,0]<0.7)/samples_mcmc['b'].shape[0])

# Test of prob b0 between 0.7 and 0.9
print('Prob of b0 in (0.7,0.9):', 
      jnp.sum((samples_mcmc['b'][:,0]>0.7)*(samples_mcmc['b'][:,0]<0.9))/samples_mcmc['b'].shape[0])
# %%
# =============================================================================
# Plate notation
# =============================================================================
# Plate notation lead to the same results as above in this case
# (exatly the same results if rng_key is the same)
# Define model
def modelPlate(X,Y=None):
    b = numpyro.sample('b', dist.Normal(0,1).expand([X.shape[1]]))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample('Y',dist.Normal(X @ b,sigma), obs=Y)
        
kernelPlate = NUTS(modelPlate)
mcmcPlate = MCMC(kernelPlate, num_samples=500, num_warmup=1000, num_chains=3)
mcmcPlate.run(rng_key_, X=X, Y=Y)
mcmcPlate.print_summary()
# %%
# =============================================================================
# Estimate same model with SVI
# =============================================================================
# Own guide for linear regression model
# FIXME this is not working yet
def own_guide_LR(X,Y=None):
    b_loc = numpyro.param('b_loc', jnp.zeros(k))
    b_scale = numpyro.param('b_scale', jnp.ones(k), constraint=dist.constraints.positive)
    sigma_loc = numpyro.param('sigma_loc', 1., constraint=dist.constraints.positive)
    b = numpyro.sample('b', dist.Normal(b_loc, b_scale).expand([k]))
    sigma = numpyro.sample('sigma', dist.Exponential(sigma_loc))

# %%
rng_key, rng_key_ = random.split(rng_key)
guide = AutoLaplaceApproximation(model)

# guide = own_guide_LR

svi = SVI(model,guide,optim.Adam(0.1),Trace_ELBO())
svi_result = svi.run(rng_key_, 1000, X=X, Y=Y)
svi_params = svi_result.params

bhat_svi = svi_params['auto_loc'][0:k]
sigma_svi = svi_params['auto_loc'][-1]
# bhat_svi = svi_params['b_loc']
# sigma_svi = svi_params['sigma_loc']

# Get sample from posterior
# FIXME this does not work yet
# See: https://github.com/pyro-ppl/numpyro/issues/1309
# predictive = Predictive(model=guide, params=svi_params, num_samples=1000)
# samples_svi = predictive(random.PRNGKey(1), X=X, Y=Y)

# Works but only for auto guide
samples_svi = guide.sample_posterior(random.PRNGKey(1), svi_params, (1000,))
numpyro.diagnostics.print_summary(samples_svi, prob=0.89, group_by_chain=False)

# %%
# Combine estimates from MCMC and SVI
lstbeta = [f'b{i}' for i in range(k)]
dfres = pd.DataFrame(np.hstack([b_true,sigma_true]).T, columns=['true'],
                   index=lstbeta+['sigma'])
dfres.loc[lstbeta,'ols'] = b_hat_ols
dfres.loc['sigma','MCMC'] = samples_mcmc['sigma'].mean(axis=0)
dfres.loc[lstbeta,'MCMC'] = samples_mcmc['b'].mean(axis=0)
dfres.loc['sigma','SVI'] = sigma_svi
dfres.loc[lstbeta,'SVI'] = bhat_svi
dfres.loc['sigma','SVI_postmean'] = samples_svi['sigma'].mean(axis=0)
dfres.loc[lstbeta,'SVI_postmean'] = samples_svi['b'].mean(axis=0)
dfres

# %%
# =============================================================================
# Model: Model perfect multicollinearity
# =============================================================================
# Estimate model using numpyro MCMC
X_multicoll = np.hstack([X,X[:,[-1]]])
print('Corr X:',pd.DataFrame(X_multicoll).corr())
#%%
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=2)
mcmc.run(rng_key_, X=X_multicoll, Y=Y)
#%%
mcmc.print_summary()
# %%
# Try this with OLS
try:
    b_hat_ols = np.linalg.inv(X_multicoll.T@X_multicoll)@X_multicoll.T@Y
    print('OLS:',b_hat_ols)
except np.linalg.LinAlgError as err:
    print('OLS failed because of: ', err)

# %%
# Calculate covariance matrix of b
samples = mcmc.get_samples()
pd.DataFrame(jnp.corrcoef(samples['b'], rowvar=False))
# => Check correlation of 3 and 4 which is close to -1

# %%
azMCMC = az.from_numpyro(mcmc)
az.plot_posterior(azMCMC, ref_val=list(b_true)+[b_true[-1],sigma_true]);

az.plot_pair(azMCMC,
            kind='kde',
            var_names=['b'],
            coords={'b_dim_0': [3,4]},
            divergences=True,
            textsize=18);
# %%
# Do the same with SVI
rng_key, rng_key_ = random.split(rng_key)
# Try out different auto guides
# guide = AutoLaplaceApproximation(model)
# Note that AutoDiagonalNormal does not capture correlation between b3 and b4
# guide = AutoDiagonalNormal(model)
guide = AutoMultivariateNormal(model)
# guide = own_guide_LR

svi = SVI(model,guide,optim.Adam(0.1),Trace_ELBO())
svi_result = svi.run(rng_key_, 1000, X=X_multicoll, Y=Y)
svi_params = svi_result.params

bhat_svi = svi_params['auto_loc'][0:k]
sigma_svi = svi_params['auto_loc'][-1]
# bhat_svi = svi_params['b_loc']
# sigma_svi = svi_params['sigma_loc']

# Get sample from posterior
# See: https://github.com/pyro-ppl/numpyro/issues/1309
# predictive = Predictive(model=guide, params=svi_params, num_samples=1000)
# samples_svi = predictive(random.PRNGKey(1), X=X_multicoll, Y=Y)
samples_svi = guide.sample_posterior(random.PRNGKey(1), svi_params, (1000,))
numpyro.diagnostics.print_summary(samples_svi, prob=0.89, group_by_chain=False)
# %%
plt.hist(samples_svi['b'][:,0],bins=50);
# %%
# Transfrom prediction data to az.inference data
s = {k:np.expand_dims(v.squeeze(),axis=0) 
        for k,v in samples_svi.items()}
azSVI = az.convert_to_inference_data(s)

az.plot_pair(azSVI,
            kind='kde',
            var_names=['b'],
            coords={'b_dim_0': [3,4]},
            divergences=True,
            textsize=18);

# %%
# =============================================================================
# Estimate linear model with real data
# =============================================================================
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
            # 'SEP_25'
            ] 


dfWheat_train = dfL_train.loc[dfL_train['crop']=='Winterweizen',:]
    
X = dfL_train.loc[dfL_train['crop']=='Winterweizen',lstColX].values 
Y = dfL_train.loc[dfL_train['crop']=='Winterweizen','yield_scaled'].values    

# Estimate model using numpyro MCMC
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=3)
mcmc.run(rng_key_, X=X, Y=Y)
mcmc.print_summary()

azMCMC = az.from_numpyro(mcmc)
azMCMC= azMCMC.assign_coords({
                'b_dim_0':lstColX,
                    })
az.summary(azMCMC)
# %%
az.plot_pair(azMCMC,
            kind='kde',
            var_names=['b'],
            coords={'b_dim_0':["MAI_25", "JUN_25", "JUL_25", "AUG_25"]},
            divergences=True,
            textsize=18);

# %%
# %%
# =============================================================================
# Logit model
# =============================================================================

# Generate artificial data
k = 4
N = 1000
X = np.random.normal(size=[N, k])*2 
X[:,0] = 1
b_true = np.arange(1,k+1)-2.5
logits = scipy.special.expit(X @ b_true)
Y = np.random.binomial(1,logits)

# Define model
def modelLogit(X,Y=None):
    b = numpyro.sample('b', dist.Normal(0,1).expand([X.shape[1]]))
    numpyro.sample('Y',dist.Bernoulli(logits=X @ b), obs=Y)
# %%
# Estimate model using numpyro MCMC
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(modelLogit)
mcmc = MCMC(kernel, num_samples=500, num_warmup=1000, num_chains=2)
mcmc.run(rng_key_, X=X, Y=Y)
mcmc.print_summary()
# Plot results using arviz
azMCMC = az.from_numpyro(mcmc)
az.plot_posterior(azMCMC, ref_val=list(b_true));

# %%
# Estimate Logit with statemodels as a comparison
log_reg = sm.Probit(Y, X).fit()
log_reg.summary()

# %%
# =============================================================================
# IV model 
# =============================================================================
# Data generation of IV model with 1 instruments and 1 endogenous variables
N = 100
k_exog = 2
exog = np.random.normal(size=(N,k_exog))
inst = np.random.normal(size=(N,1))
unobs = np.random.normal(size=(N,1))

a_inst = 0.1
a_unobs = -0.5
endog = a_inst * inst + a_unobs * unobs #+ np.random.normal(0,1,size=(N,1))

b_exog = np.array(np.arange(k_exog)).reshape(-1,1)
b_endog = -1
b_unobs = -1
sigma = 0.1
y = exog @ b_exog +b_endog * endog +  b_unobs * unobs + np.random.normal(0,sigma,size=(N,1))
assert y.shape == (N,1)


X = np.hstack([exog,endog])
bOls = np.linalg.inv(X.T@X)@X.T@y
print('beta OLS manual\n', bOls)
# Verify with manual 2SLS (see Verbeek 2008 p. 153)
Z = np.hstack([exog,inst])
invZZ = np.linalg.inv(Z.T@Z)
XZ = X.T@Z
ZX = Z.T@X
bIV = np.linalg.inv(XZ@invZZ@ZX)@XZ@invZZ@Z.T@y
print('beta 2sls manual\n',bIV)


dfres = pd.DataFrame(np.vstack([b_exog,b_endog]), columns=['true'],
                   index=['b_exog_0','b_exog_1','b_endog'])
dfres['OLS'] = bOls
dfres['2slsManual'] = bIV
dfres

# Use statsmodels IV2SLS
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
mod = IV2SLS(y, np.hstack([exog,endog]), instrument=np.hstack([exog,inst]))    # Describe model
res = mod.fit()  
print(res.summary()) 
# get coefficients
res.params
dfres['2sls'] = res.params
dfres

# %%
# Build model with numpyro
# Based on Statistical rethinking 2nd edition section 14.3.1
# and https://github.com/fehiepsi/rethinking-numpyro 
def model(exog, endog, inst,y=None):
    
    b_exog = numpyro.sample("b_exog", dist.Normal(0, 0.5).expand([exog.shape[1],1]))
    b_endog = numpyro.sample("b_endog", dist.Normal(0, 0.5))
    a_inst = numpyro.sample("a_inst", dist.Normal(0, 0.5))
    
    muY = exog @ b_exog + b_endog * endog
    muEndog = a_inst * inst
    
    Rho = numpyro.sample("Rho", dist.LKJ(2, 2))
    Sigma = numpyro.sample("Sigma", dist.Exponential(1).expand([2]))
    
    cov = jnp.outer(Sigma, Sigma) * Rho
    
    numpyro.sample(
        "y,endog",
        dist.MultivariateNormal(jnp.stack([muY, muEndog], -1), cov),
        obs=jnp.stack([y, endog], -1),
    )


mcmc = MCMC(NUTS(model), num_warmup=800, num_samples=500, num_chains=3)
mcmc.run(random.PRNGKey(0), exog, endog, inst,y=y)
mcmc.print_summary(0.89)


samples = mcmc.get_samples()
dfres['PP'] = np.vstack([samples['b_exog'].mean(axis=0),
                         samples['b_endog'].mean()])
dfres
# %%
Rho = samples['Rho'].mean(axis=0)
Sigma = samples['Sigma'].mean(axis=0)
jnp.outer(Sigma, Sigma) * Rho

# %%
# =============================================================================
# Model 1: Model with random intercepts and slopes
# =============================================================================
# =============================================================================
# Model 1: Model with random intercepts and slopes
# =============================================================================
def model_M1(soil,smi,yield_crop=None):

    # Prior on beta coefficients    
    betaSoil = numpyro.sample('betaSoil',dist.Normal(0,0.5))
    betaSmi = numpyro.sample('betaSmi',
                               dist.Normal(0,0.5).expand([smi.shape[1]]))
    # Prior on sigma
    sigma_yield = numpyro.sample('sigma_yield',dist.Exponential(2))
    
    # linear model X*beta
    mu = betaSoil*soil + jnp.sum(betaSmi * smi,axis=1)

    numpyro.sample('yield',dist.Normal(mu,sigma_yield),obs=yield_crop)


    
# %%
# =============================================================================
# Model 2: Model with random intercepts and slopes
# =============================================================================
def model_M2(soil,smi = None, smiDiff = None,yield_crop=None):

    # Prior on beta coefficients    
    betaSoil = numpyro.sample('betaSoil',dist.Normal(0,0.5))
    
    Xsmi, XsmiDiff = 0., 0.
    if smi is not None:
        betaSmi = numpyro.sample('betaSmi',
                                dist.Normal(0,0.5).expand([smi.shape[1]]))
        Xsmi =  jnp.sum(betaSmi * smi,axis=1)
    
    if smiDiff is not None:
        betaSmiDiff = numpyro.sample('betaSmiDiff',
                                dist.Normal(0,0.5).expand([smiDiff.shape[1]]))
        XsmiDiff =  jnp.sum(betaSmiDiff * smiDiff,axis=1)
        
    # Prior on sigma
    sigma_yield = numpyro.sample('sigma_yield',dist.Exponential(2))
    
    # linear model X*beta
    mu = betaSoil*soil + Xsmi + XsmiDiff

    numpyro.sample('yield',dist.Normal(mu,sigma_yield),obs=yield_crop)
# %%
rng_key = random.PRNGKey(0)

dfWheat_train = dfL_train.loc[dfL_train['crop']=='Winterweizen',:]

data = dict(soil=dfWheat_train['bodenzahl_scaled'].values,
            smi=dfWheat_train[lstSmiMonth].values,
            # smi_diff=dfWheat_train[lstSmiDiff].values,
            yield_crop=dfWheat_train['yield_scaled'].values)


rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model_M1)
mcmcM1 = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=3)
mcmcM1.run(rng_key_, **data)

# Print summary
mcmcM1.print_summary()
# Get samples
samples = mcmcM1.get_samples()

# %%
# Transform samples to arviz format
azMCMC_M1 = az.from_numpyro(mcmcM1)
# azMCMC= azMCMC.assign_coords({'betaSmi_dim_0':lstSmiMonth})
azMCMC_M1= azMCMC_M1.assign_coords({
                            'betaSmi_dim_0':lstSmiMonth,
                              })
az.plot_forest(azMCMC_M1,combined=True, ridgeplot_overlap=1);
# %%
# =============================================================================
# Model 2
# =============================================================================
data2 = dict(soil=dfWheat_train['bodenzahl_scaled'].values,
            smi=dfWheat_train[lstSmiMonth].values,
            smiDiff=dfWheat_train[lstSmiDiff].values,
            yield_crop=dfWheat_train['yield_scaled'].values)


rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model_M2)
mcmcM2 = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=3)
mcmc.run(rng_key_, **data2)

# Print summary
mcmc.print_summary()
# Get samples
samples = mcmc.get_samples()

# %%
# Transform samples to arviz format
azMCMC = az.from_numpyro(mcmc)
# azMCMC= azMCMC.assign_coords({'betaSmi_dim_0':lstSmiMonth})
azMCMC= azMCMC.assign_coords({
                            'betaSmi_dim_0':lstSmiMonth,
                            'betaSmiDiff_dim_0':lstSmiDiff,
                              })
az.plot_forest(azMCMC,combined=True, ridgeplot_overlap=1);
# %%
# %%
# rename columns
dfWeizen
# %%
import seaborn as sns
sns.pairplot(df, x_vars=[strY], y_vars=[strY]+lstX);
# %%
sns.pairplot(df, x_vars=[strY], y_vars=lstX);
# %%
sns.pairplot(dfWeizen, x_vars=[strY], y_vars=lstSmiMonthDiff);
# %%
sns.pairplot(df, x_vars=lstX, y_vars=lstX);
# %%
dfWeizen['MAI_25'].hist(bins=100)