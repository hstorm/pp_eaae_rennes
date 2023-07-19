# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import statsmodels.api as sm

import jax.numpy as jnp
from jax import nn, lax, random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
# numpyro.set_platform("gpu")
numpyro.set_host_device_count(3)

# %%
rng_key = random.PRNGKey(1)

# %%
# Example market model 
def modelMarket(quant=None,price=None,demandShocks=None, supplyShocks=None):
    
    demandShocks = numpyro.sample("demandShocks", dist.Normal(0, 1.), obs=demandShocks)
    supplyShocks = numpyro.sample("supplyShocks", dist.Normal(0, 1.), obs=supplyShocks)
    
    bPrice_supply = numpyro.sample("bPrice_supply", 
                                   dist.TruncatedNormal(-0.5, 0.5, low=0),
                                   )
    bPrice_demand = numpyro.sample("bPrice_demand", 
                                   dist.TruncatedNormal(-0.5, 0.5, high=0),
                                   )
    
    bCons_demand = numpyro.sample("bCons_demand", dist.TruncatedNormal(2, 1,low=0))
    bCons_supply = numpyro.sample("bCons_supply", dist.TruncatedNormal(0, 1,high=0))
    
    bDemandShock = numpyro.sample("bDemandShock", dist.Normal(0, 0.5))
    bSupplyShock = numpyro.sample("bSupplyShock", dist.Normal(0, 0.5))
    
    price_mu = (bCons_demand
                +bDemandShock * demandShocks 
                - bCons_supply
                - bSupplyShock * supplyShocks )/(bPrice_supply-bPrice_demand)
    
    if price is not None:
        price  = numpyro.sample("price", dist.Normal(price_mu, 0.5), obs=price)
    else: 
        price = numpyro.deterministic('price',price_mu)
    
    mu_quant_demand = jnp.clip(bCons_demand + bPrice_demand * price + bDemandShock * demandShocks,0,np.inf)
    mu_quant_supply = jnp.clip(bCons_supply + bPrice_supply * price + bSupplyShock * supplyShocks,0,np.inf)
    # mu_quant_demand = bCons_demand + bPrice_demand * price + bDemandShock * demandShocks
    # mu_quant_supply = bCons_supply + bPrice_supply * price + bSupplyShock * supplyShocks

    if quant is not None:
        numpyro.sample("quant_demand", 
                        dist.Normal(mu_quant_demand, 0.5), obs=quant)
        numpyro.sample("quant_supply",
                        dist.Normal(mu_quant_supply, 0.5), obs=quant)
    else:
        numpyro.deterministic('quant_demand',mu_quant_demand)
        numpyro.deterministic('quant_supply',mu_quant_demand)

# ===========================
# Use model to generate data
# ===========================
# True parameter values
bPrice_demand_true =  -0.5   
bPrice_supply_true =  0.3   
bCons_demand_true =  3   
bCons_supply_true =  -1 
bDemandShock_true = 1
bSupplyShock_true = 1 

# Condition the model
condition_model = numpyro.handlers.condition(modelMarket, data={
                            'bPrice_demand':bPrice_demand_true,
                            'bPrice_supply':bPrice_supply_true,
                            'bCons_demand':bCons_demand_true,
                            'bCons_supply':bCons_supply_true,
                            'bDemandShock':bDemandShock_true,
                            'bSupplyShock':bSupplyShock_true,
                                })
# Sample data
nPriorSamples = 500
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(condition_model, num_samples=nPriorSamples)
prior_samples = prior_predictive(rng_key_)

assert np.unique(prior_samples['bCons_demand']) == bCons_demand_true
assert np.array_equal(prior_samples['quant_demand'],prior_samples['quant_supply'])

# Get Data required for inference
price = prior_samples['price']
quant = prior_samples['quant_demand']
supplyShocks = prior_samples['supplyShocks']
demandShocks= prior_samples['demandShocks']
# %%
# Plot generated data
nPlotSamples = 50 # number of sample to plot
xPlot = np.linspace(0,15,15).reshape(1,-1)
yDemand = (
            prior_samples['bCons_demand'].reshape(-1,1)+
           prior_samples['bPrice_demand'].reshape(-1,1)*xPlot
           +(prior_samples['bDemandShock'] * prior_samples['demandShocks']).reshape(-1,1)
           )
ySupply = (prior_samples['bCons_supply'].reshape(-1,1)+
            prior_samples['bPrice_supply'].reshape(-1,1)*xPlot
           +(prior_samples['bSupplyShock'] * prior_samples['supplyShocks']).reshape(-1,1)
           )
fig, ax = plt.subplots();
color = iter(cm.rainbow(np.linspace(0, 1, nPlotSamples)))
for i in range(nPlotSamples):
    c = next(color)
    ax.plot(xPlot.T,yDemand[i,:],alpha=0.4,color=c)
    ax.plot(xPlot.T,ySupply[i,:],alpha=0.4,color=c)
    ax.scatter(prior_samples['price'][:nPlotSamples],prior_samples['quant_demand'][:nPlotSamples],color=c)
ax.set_ylim(0,5);
ax.set_xlim(0,15);

# %%
# ==========================
# Inference
# ==========================
mcmc = MCMC(NUTS(modelMarket), num_warmup=800, num_samples=500, num_chains=3)
mcmc.run(random.PRNGKey(0), price=price, 
         quant=quant, 
         supplyShocks=supplyShocks,
         demandShocks=demandShocks)
mcmc.print_summary(0.89)

samples = mcmc.get_samples()

# Plot results
azMCMC = az.from_numpyro(mcmc)
az.plot_posterior(azMCMC, var_names=['bCons_demand','bPrice_demand','bCons_supply','bPrice_supply'],
                  ref_val= [bCons_demand_true,bPrice_demand_true,bCons_supply_true,bPrice_supply_true]);

# %%
# ==========================
# Compare to manual Two-Stage least squares
# ==========================
Y1 = price.reshape(-1,1)
X1 = np.concatenate([np.ones([Y1.shape[0],1]),supplyShocks.reshape(-1,1),demandShocks.reshape(-1,1)],axis=1)
b_hat_ols1 = np.linalg.inv(X1.T@X1)@X1.T@Y1
print('OLS 1th stage: \n',b_hat_ols1)
priceHat = X1@b_hat_ols1

Y2 = quant.reshape(-1,1)
X2demand = np.concatenate([np.ones([Y1.shape[0],1]),priceHat,demandShocks.reshape(-1,1)],axis=1)
X2supply = np.concatenate([np.ones([Y1.shape[0],1]),priceHat,supplyShocks.reshape(-1,1)],axis=1)
b_hat_ols2Demand = np.linalg.inv(X2demand.T@X2demand)@X2demand.T@Y2
b_hat_ols2Supply = np.linalg.inv(X2supply.T@X2supply)@X2supply.T@Y2

resDemand = pd.DataFrame(b_hat_ols2Demand,index=['cons','bprice','bDemandShock'],columns=['2SLS'])
resDemand['true'] = [bCons_demand_true,bPrice_demand_true,bDemandShock_true]
print(resDemand)
resSupply = pd.DataFrame(b_hat_ols2Supply,index=['cons','bprice','bSupplyShock'],columns=['2SLS'])
resSupply['true'] = [bCons_supply_true,bPrice_supply_true,bSupplyShock_true]
print(resSupply)

# %%
