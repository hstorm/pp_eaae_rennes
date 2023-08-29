# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

# Adjust to own setting (correct for VS code devcontainer)
os.chdir("/workspaces/pp_eaae_rennes/")
# %%
rng_key = random.PRNGKey(1)

# %%
# =============================================================================
# Define most basic linear regression model
# =============================================================================
def model():
    x = numpyro.sample('x', dist.Normal(0,1))


# =============================================================================
# Prior sampling
# =============================================================================
sigma_b = 3
nSamples = 300
# Prior sampling
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(model, num_samples=nSamples)
prior_samples = prior_predictive(rng_key_)
# %%

x_samples = prior_samples['x']

cum_sum= np.cumsum(x_samples, axis=0)
cum_mean = cum_sum / np.arange(1, nSamples+1)

# %%
xlim = 300
# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
# ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# no labels
ax_histy.tick_params(axis="y", labelleft=False)

import scipy.stats as stats
import math
mu = 0
sigma = 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax_histy.plot(stats.norm.pdf(x, mu, sigma),x, color='grey', linestyle='dashed', linewidth=1)

ax.set_ylabel('E[X]')
ax.set_xlabel('Number of samples')
# the line plot:
line, = ax.plot(np.arange(1,nSamples+1),cum_mean[:nSamples])
point = ax.scatter(0,x_samples[nSamples], color='red', s=1)
# 
# now determine nice limits by hand:
binwidth = 0.25
xymax = x_samples.max()
lim = (int(xymax/binwidth) + 1) * binwidth

bins = np.arange(-lim, lim + binwidth, binwidth)
_, _, bar_container = ax_histy.hist(x_samples,bins=bins,
                                    orientation='horizontal',
                                    density=True)


def prepare_animation(bar_container):
    def animate(i):
        line.set_data(np.arange(1,i+1),cum_mean[:i])
        point.set_offsets((0,x_samples[i]))
        # point.set_data(0,x_samples[i])
        # simulate new data coming in
        data = x_samples[:i]
        n, _ = np.histogram(data, bins=bins, density=True)
        for count, rect in zip(n, bar_container.patches):
            rect.set_width(count)
        return bar_container.patches
    return animate

# call the animator.  blit=True means only re-draw the parts that have changed.
ani = animation.FuncAnimation(fig, prepare_animation(bar_container),
                               frames=xlim, interval=20, blit=True)

# # To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save(os.path.join('illustrations','scatter.gif'), writer=writer)

plt.show()




