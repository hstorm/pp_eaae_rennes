# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
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
# numpyro.set_platform("cpu")
numpyro.set_platform("gpu")
numpyro.set_host_device_count(2)

# %%
rng_key = random.PRNGKey(2)

# %%
# Example market model 
def modelMarket_old(N, quant=None,price=None,demandShocks=None, supplyShocks=None,
                latDemandShift=None,latSupplyShift=None):
    
    demandShocks = numpyro.sample("demandShocks", dist.Normal(0, 1.).expand([N,]), obs=demandShocks)
    supplyShocks = numpyro.sample("supplyShocks", dist.Normal(0, 1.).expand([N,]), obs=supplyShocks)
    
    bPrice_supply = numpyro.sample("bPrice_supply", dist.TruncatedNormal(0, 5, low=0))
    bPrice_demand = numpyro.sample("bPrice_demand",dist.TruncatedNormal(0, 5, high=0))
    
    bCons_demand = numpyro.sample("bCons_demand", dist.TruncatedNormal(1, 1,low=0))
    bCons_supply = numpyro.sample("bCons_supply", dist.TruncatedNormal(0, 1,high=0))
    
    bDemandShock = numpyro.sample("bDemandShock", dist.Normal(0, 5))
    bSupplyShock = numpyro.sample("bSupplyShock", dist.Normal(0, 5))
    
    # mu_latDemandShift = numpyro.sample("mu_latDemandShift", dist.Normal(0, 0.5))
    # sigma_latDemandShift = numpyro.sample("sigma_latDemandShift", dist.HalfNormal(0.5))
    # mu_latSupplyShift = numpyro.sample("mu_latSupplyShift", dist.Normal(0, 0.5))
    # sigma_latSupplyShift = numpyro.sample("sigma_latSupplyShift", dist.HalfNormal(0.5))
    
    with numpyro.plate("data", N):
        # latDemandShift = numpyro.sample("latDemandShift", dist.Normal(mu_latDemandShift, sigma_latDemandShift).expand([1,]))
        # latSupplyShift = numpyro.sample("latSupplyShift", dist.Normal(mu_latSupplyShift, sigma_latSupplyShift).expand([1,]))
        latDemandShift = numpyro.sample("latDemandShift", dist.Normal(0., 0.5).expand([1,]), obs=latDemandShift)
        latSupplyShift = numpyro.sample("latSupplyShift", dist.Normal(0., 0.5).expand([1,]), obs=latSupplyShift)

        a = jnp.array([[-1, 0, bPrice_supply],[0, -1, bPrice_demand],[1,-1,0]])
        b = jnp.array([- bSupplyShock * supplyShocks - latSupplyShift,
                    -bDemandShock * demandShocks - latDemandShift,
                    jnp.zeros_like(supplyShocks)])

        x = jnp.linalg.solve(a, b)
        quant = x[0]
        price = x[2]
        numpyro.deterministic('quant_supply',x[0])
        numpyro.deterministic('quant_demand',x[1])
        
        
    if price is not None:
        price  = numpyro.sample("price", dist.Normal(price, 0.3), obs=price)
    else: 
        price = numpyro.deterministic('price',price)
    if quant is not None:
        numpyro.sample("quant",dist.Normal(quant, 0.5), obs=quant)
    else:
        numpyro.deterministic('quant',quant)
    # %
    # with numpyro.plate("data", N):
        
    #     latDemandShift = numpyro.deterministic('latDemandShift',0)
    #     latSupplyShift = numpyro.deterministic('latSupplyShift',0)
    #     # latDemandShift = numpyro.sample("latDemandShift", dist.Normal(mu_latDemandShift, sigma_latDemandShift))
    #     # latSupplyShift = numpyro.sample("latSupplyShift", dist.Normal(mu_latSupplyShift, sigma_latSupplyShift))
        
    #     price_mu = (bCons_demand+bDemandShock * demandShocks + latDemandShift
    #                 - bCons_supply - bSupplyShock * supplyShocks - latSupplyShift)/(bPrice_supply-bPrice_demand)
        
    #     if price is not None:
    #         price  = numpyro.sample("price", dist.Normal(price_mu, 0.3), obs=price)
    #     else: 
    #         price = numpyro.deterministic('price',price_mu)
        
    #     quant = (((bCons_demand + bDemandShock * demandShocks + latDemandShift)*bPrice_supply-
    #             (bCons_supply + bSupplyShock * supplyShocks + latSupplyShift)*bPrice_demand)/(bPrice_supply-bPrice_demand))
        
    #     quant = jnp.clip(quant,0,np.inf)    
        
    #     # mu_quant_demand = jnp.clip(bCons_demand + bPrice_demand * price + bDemandShock * demandShocks,0,np.inf)
    #     # mu_quant_supply = jnp.clip(bCons_supply + bPrice_supply * price + bSupplyShock * supplyShocks,0,np.inf)
    #     # mu_quant_demand = bCons_demand + bPrice_demand * price + bDemandShock * demandShocks + latDemandShift
    #     # mu_quant_supply = bCons_supply + bPrice_supply * price + bSupplyShock * supplyShocks + latSupplyShift

    #     if quant is not None:

    #         numpyro.sample("quant",
    #                         dist.Normal(quant, 0.5), obs=quant)

    #         # numpyro.sample("quant_supply",
    #         #                 dist.Normal(mu_quant_supply, 0.5), obs=quant)
    #         # numpyro.sample("quant_demand", 
    #         #                 dist.Normal(mu_quant_demand, 0.5), obs=quant)
    #     else:
    #         # numpyro.deterministic('quant',mu_quant_demand)
    #         numpyro.deterministic('quant',quant)
    #         # numpyro.deterministic('quant_supply',quant)
# %%



# %%
# Example market model 
def modelMarket(N, quant=None,price=None,demandShocks=None, supplyShocks=None):
    
    demandShocks = numpyro.sample("demandShocks", dist.Normal(0, 1.).expand([N,]), obs=demandShocks)
    supplyShocks = numpyro.sample("supplyShocks", dist.Normal(0, 1.).expand([N,]), obs=supplyShocks)
    
    bCons_demand = numpyro.sample("bCons_demand", dist.TruncatedNormal(1, 1,low=0))
    bCons_supply = numpyro.sample("bCons_supply", dist.TruncatedNormal(0, 1,high=0))
    
    bPrice_supply = numpyro.sample("bPrice_supply", dist.TruncatedNormal(0, 1, low=0))
    bPrice_demand = numpyro.sample("bPrice_demand",dist.TruncatedNormal(0, 1, high=0))
   
    bDemandShock = numpyro.sample("bDemandShock", dist.Normal(0, 1))
    bSupplyShock = numpyro.sample("bSupplyShock", dist.Normal(0, 1))

    # Solve a linear system of equations of the from: 
    #   A*x=B 
    #   with
    #   [Q_s,Q_d,P]^T = x    
    A = jnp.array([ [-1, 0, bPrice_supply],
                    [0, -1, bPrice_demand],
                    [1,-1,0]])
    B = jnp.array([
                -bCons_supply - bSupplyShock * supplyShocks ,
                -bCons_demand - bDemandShock * demandShocks ,
                jnp.zeros_like(supplyShocks)
                ])
    x = jnp.linalg.solve(A, B)

    numpyro.sample("quant",dist.LogNormal(x[0,:], 0.005), obs=quant)
    numpyro.sample("price",dist.LogNormal(x[2,:], 0.005), obs=price)

        
# ===========================
# Use model to generate data
# ===========================
# True parameter values
bPrice_demand_true =  -0.9   
bPrice_supply_true =  0.8   
bCons_demand_true =  1   
bCons_supply_true =  -1 
bDemandShock_true = 0.5
bSupplyShock_true = 1 

# Condition the model
condition_model = numpyro.handlers.condition(modelMarket, data={
                            'bPrice_demand':bPrice_demand_true,
                            'bPrice_supply':bPrice_supply_true,
                            'bCons_demand':bCons_demand_true,
                            'bCons_supply':bCons_supply_true,
                            # 'mu_latDemandShift':bCons_demand_true,
                            # 'mu_latSupplyShift':bCons_supply_true,
                            'bDemandShock':bDemandShock_true,
                            'bSupplyShock':bSupplyShock_true,
                                })
# Sample data
nPriorSamples = 500
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(condition_model, num_samples=nPriorSamples)
prior_samples = prior_predictive(rng_key_, N=1)

assert np.unique(prior_samples['bPrice_demand']) == bPrice_demand_true
# assert np.unique(prior_samples['mu_latDemandShift']) == bCons_demand_true
# assert np.allclose(np.round(prior_samples['quant_demand'],1),np.round(prior_samples['quant_supply'],1))

# Get Data required for inference
price = prior_samples['price'].squeeze()
quant = prior_samples['quant'].squeeze()
# price = prior_samples['price,quant'][:,:,0].squeeze()
# quant = prior_samples['price,quant'][:,:,1].squeeze()
supplyShocks = prior_samples['supplyShocks'].squeeze()
demandShocks= prior_samples['demandShocks'].squeeze()
quant.max()
# %%
# Plot generated data
nPlotSamples = 5 # number of sample to plot
xPlot = np.linspace(0,15,15).reshape(1,-1)
yDemand = jnp.exp(
            prior_samples['bCons_demand'].reshape(-1,1)+
           prior_samples['bPrice_demand'].reshape(-1,1)*xPlot
           +(prior_samples['bDemandShock'] * demandShocks).reshape(-1,1)
        #    +prior_samples['latDemandShift'].reshape(-1,1)
           )
ySupply = jnp.exp(
            prior_samples['bCons_supply'].reshape(-1,1)+
            prior_samples['bPrice_supply'].reshape(-1,1)*xPlot
           +(prior_samples['bSupplyShock'] * supplyShocks).reshape(-1,1)
        #    +prior_samples['latSupplyShift'].reshape(-1,1)
           )
fig, ax = plt.subplots();
color = iter(cm.rainbow(np.linspace(0, 1, nPlotSamples)))
for i in range(nPlotSamples):
    c = next(color)
    ax.plot(xPlot.T,yDemand[i,:],alpha=0.4,color=c)
    ax.plot(xPlot.T,ySupply[i,:],alpha=0.4,color=c)
    ax.scatter(jnp.log(price[:nPlotSamples]),quant[:nPlotSamples],color=c)
ax.set_ylim(0,3);
ax.set_xlim(0,5);

# %%
# ==========================
# Inference
# ==========================
mcmc = MCMC(NUTS(modelMarket), num_warmup=800, num_samples=500, num_chains=2)
mcmc.run(random.PRNGKey(0),N=price.shape[0],price=price, 
         quant=quant, 
        #  supplyShocks=supplyShocks,
         demandShocks=demandShocks
         )
mcmc.print_summary(0.89)

samples = mcmc.get_samples()
# %%
# Plot results
azMCMC = az.from_numpyro(mcmc)
az.plot_posterior(azMCMC, var_names=[
                            'bCons_demand',
                            'bPrice_demand',
                            'bDemandShock',
                            'bCons_supply',
                            'bPrice_supply',
                            'bSupplyShock'],
                  ref_val= [
                            bCons_demand_true,
                            bPrice_demand_true,
                            bDemandShock_true,
                            bCons_supply_true,
                            bPrice_supply_true,
                            bSupplyShock_true
                            ]);

# %%
quant = quant.reshape(-1,1)
price = price.reshape(-1,1)
supplyShocks = supplyShocks.reshape(-1,1)
demandShocks = demandShocks.reshape(-1,1)

# Prepare res data frame
dfres = pd.DataFrame(np.vstack([bCons_demand_true,
                                bPrice_demand_true,
                                bDemandShock_true,
                                bCons_supply_true,
                                bPrice_supply_true,
                                bSupplyShock_true]), 
                        columns=['true'],
                        index=['bCons_demand',
                                'bPrice_demand',
                                'bDemandShock',
                                'bCons_supply',
                                'bPrice_supply',
                                'bSupplyShock'])

dfres['DGP PP'] = np.vstack([samples['bCons_demand'].mean(),
                        samples['bPrice_demand'].mean(),
                        samples['bDemandShock'].mean(),
                        samples['bCons_supply'].mean(),
                        samples['bPrice_supply'].mean(),
                        samples['bSupplyShock'].mean()])
                    
dfres


# %%
# ==========================
# Compare to manual Two-Stage least squares
# ==========================
Y1 = price
X1 = np.concatenate([np.ones([Y1.shape[0],1]),supplyShocks,demandShocks],axis=1)
b_hat_ols1 = np.linalg.inv(X1.T@X1)@X1.T@Y1
print('OLS 1th stage: \n',b_hat_ols1)
priceHat = X1@b_hat_ols1

# calculate r2
from sklearn.metrics import r2_score
print('R2 1th Stage',r2_score(Y1,priceHat))

Y2 = quant
X2demand = np.concatenate([np.ones([Y1.shape[0],1]),priceHat,demandShocks],axis=1)
X2supply = np.concatenate([np.ones([Y1.shape[0],1]),priceHat,supplyShocks],axis=1)
b_hat_ols2Demand = np.linalg.inv(X2demand.T@X2demand)@X2demand.T@Y2
b_hat_ols2Supply = np.linalg.inv(X2supply.T@X2supply)@X2supply.T@Y2

dfres['OLS 2SLS manual'] = np.vstack([b_hat_ols2Demand,
                                      b_hat_ols2Supply])
dfres



# %%
# Verify with manual 2SLS (see Verbeek 2008 p. 153)
N = price.shape[0]
inst = Z = np.hstack([np.ones([N,1]),supplyShocks,demandShocks])
endog = price
exogSupply = np.hstack([np.ones([N,1]),supplyShocks])
exogDemand = np.hstack([np.ones([N,1]),demandShocks])

Xsupply = np.hstack([np.ones([N,1]),endog,supplyShocks])
Xdemand = np.hstack([np.ones([N,1]),endog,demandShocks])
y = np.array(quant)
invZZ = np.linalg.inv(Z.T@Z)
XZsupply = Xsupply.T@Z
XZdemand = Xdemand.T@Z
ZXsupply = Z.T@Xsupply
ZXdemand = Z.T@Xdemand
bIVsupply = np.linalg.inv(XZsupply@invZZ@ZXsupply)@XZsupply@invZZ@Z.T@y
bIVdemand = np.linalg.inv(XZdemand@invZZ@ZXdemand)@XZdemand@invZZ@Z.T@y

dfres['IV 2SLS matrix'] = np.vstack([bIVdemand,
                                     bIVsupply])
dfres

# %%
# Use statsmodels IV2SLS
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
modSupply= IV2SLS(y, Xsupply, instrument=Z)    
modDemand= IV2SLS(y, Xdemand, instrument=Z)    
resSupply = modSupply.fit()  
resDemand = modDemand.fit()  

dfres['IV 2SLS statsmodels'] = np.vstack([resDemand.params.reshape(-1,1),
                                          resSupply.params.reshape(-1,1)])
dfres
# %%
# Build model with numpyro
# Based on Statistical rethinking 2nd edition section 14.3.1
# and https://github.com/fehiepsi/rethinking-numpyro 
def model(exog, endog, inst,y=None):
    
    b_exog = numpyro.sample("b_exog", dist.Normal(0, 0.5).expand([exog.shape[1],1]))
    b_endog = numpyro.sample("b_endog", dist.Normal(0, 0.5))
    a_inst = numpyro.sample("a_inst", dist.Normal(0, 0.5).expand([inst.shape[1],1]))
    
    muY = exog @ b_exog + b_endog * endog
    muEndog = inst @ a_inst # inst matrix needs to include instruments and exogenous variables
    
    Rho = numpyro.sample("Rho", dist.LKJ(2, 2))
    Sigma = numpyro.sample("Sigma", dist.Exponential(1).expand([2]))
    
    cov = jnp.outer(Sigma, Sigma) * Rho
    
    numpyro.sample(
        "y,endog",
        dist.MultivariateNormal(jnp.stack([muY, muEndog], -1), cov),
        obs=jnp.stack([y, endog], -1),
    )


mcmcSupply = MCMC(NUTS(model), num_warmup=800, num_samples=500, num_chains=2)
mcmcSupply.run(random.PRNGKey(0), exog=exogSupply, endog=endog, inst=inst, y=y)
mcmcSupply.print_summary(0.89)
mcmcDemand = MCMC(NUTS(model), num_warmup=800, num_samples=500, num_chains=2)
mcmcDemand.run(random.PRNGKey(0), exog=exogDemand, endog=endog, inst=inst, y=y)
mcmcDemand.print_summary(0.89)

samplesSupply = mcmcSupply.get_samples()
samplesDemand = mcmcDemand.get_samples()
# %%
b_exog_supply_mean = samplesSupply['b_exog'].mean(axis=0)
b_exog_demand_mean = samplesDemand['b_exog'].mean(axis=0)
dfres['IV PP'] = np.vstack([
                        b_exog_demand_mean[0,0],
                        samplesDemand['b_endog'].mean(),
                        b_exog_demand_mean[1,0],
                        b_exog_supply_mean[0,0],
                        samplesSupply['b_endog'].mean(),
                        b_exog_supply_mean[1,0],
                        ])
dfres
# %%
