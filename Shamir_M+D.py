# Calculate parameter constraints and create corner plot in a Bayesian MCMC analysis
# This code infers monopole, dipole and quadrupole, but can easily be modified to restrict to some subset
# Parallelised Open-MP in MCMC sampling

import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt
import corner
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import sys
import os
os.makedirs("Plots", exist_ok=True)


data_choice = 5

if data_choice==0: filename = 'Longo.csv'
elif data_choice==1: filename = 'Iye.csv'
elif data_choice==2: filename = 'SDSS_DR7.csv'
elif data_choice==3: filename = 'GAN_M.csv'
elif data_choice==4: filename = 'GAN_NM.csv'
elif data_choice==5: filename = 'Shamir.csv'
elif data_choice==6: filename = 'PS_DR1.csv'
else:
    print("Wrong dataset choice")
    quit()

print("Analysing dataset", filename)

name = filename[:-4]

data = pd.read_csv(r"/media/vedant/CE80C1D280C1C0E7/Users/DELL/Desktop/astro/Anisotropy/Datasets/Shamir.csv")

if filename == 'Longo.csv':
    data_low_z = data[(data['RS'] < 0.04) & (data['G'] < 17)]
    print("Number of galaxies after G < 17 cut for z < 0.04:", len(data_low_z))

    data_high_z = data[(data['RS'] >= 0.04) & (data['G'] < 17.4)]
    print("Number of galaxies after G < 17.4 cut for z >= 0.04:", len(data_high_z))

    final_data = data[(data['G'] <= 17.4) & (data['RS'] <= 0.085)]
    print("Number of galaxies after G <= 17.4 and z <= 0.085 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) > 1.6]
    print("Number of galaxies after (U - Z) > 1.6 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) < 3.5]
    print("Number of galaxies after (U - Z) < 3.5 cut:", len(final_data))

    data = final_data

    data['Spin'] = data['Spin'].apply(lambda x: 1 if x == 'L' else 0)
    cw_ccw = data['Spin'].values
else:
    data['cw'] = data['cw'].apply(lambda x: 1 if x == 1 else 0)
    cw_ccw = data['cw'].values

RA = np.radians(data['ra'])
Dec = np.radians(data['dec'])
n_ra = np.radians(data['ra'])
n_dec = np.radians(data['dec'])
n_theta = np.pi/2 - n_dec

spin = np.array(cw_ccw)

N_CW = np.sum(spin)
N_tot = len(spin)

print("Mean spin:", np.mean(spin))

def lnlike(x,*args):

    M = x[0]
    D, d_ra, d_dec= x[1], x[2], x[3]

    if M<0.3 or M>0.7 or D>0.3 or D<0. :
        return -np.inf

    if d_dec>np.pi/2 or d_dec<-np.pi/2 or d_ra>2.*np.pi or d_ra<0:
        return -np.inf

    

    d_theta = np.pi/2 - d_dec
    

    #(sinθ1sinθ2cos(ϕ1−ϕ2)+cosθ1cosθ2)

    prob = M + D * (np.sin(d_theta) * np.sin(n_theta) * np.cos(d_ra - n_ra) + np.cos(d_theta) * np.cos(n_theta)) 
    if np.sum(prob<0)>0 or np.sum(prob>1)>0:
        print("Sampling P<0 or P>1:", M, D, Q, flush=True)
        return -np.inf

    lnlike = np.sum(np.log(prob)*spin + np.log(1-prob)*(1-spin))
    
    return lnlike

def lnprob(x,*args):
    d_dec = x[3]

    if d_dec>np.pi/2 or d_dec<-np.pi/2:
        return -np.inf

    

    lnlike_value = lnlike(x,*args)

    lnprior = np.log(np.cos(d_dec)) 
    return lnlike_value + lnprior

def nll(x):
    return -lnlike(x)

min_values = []
min1 = np.inf

for m in range(10):
    inpt = [np.random.uniform(0.4, 0.6), np.random.uniform(0., 0.2), np.random.uniform(0, 2.*np.pi), np.random.uniform(-np.pi/2, np.pi/2)]
    
    
    res = minimize(nll, inpt, method="Nelder-Mead")
    if res['fun'] < min1:
        params = res.x
        min1 = res['fun']
    min_values.append("Iteration {} - Min value: {}".format(m+1, min1))

print("ML params:", params, min1)
# print("Min values:")
# for value in min_values:
    # print(value)

L = lnlike(params)

k = len(params)
n = len(data)

bic = -2 * (L) + k * np.log(n)

print("BIC:", bic)

bic_values_for_M = [
    21012.4110286252, # Longo
    101051.90731787,  # Iye
    8469.24333237917, # SDSS_DR7
    193871.969825225, # GAN_M
    192618.331159577, # GAN_NM
    107916.724775013, # Shamir
    39828.6986758339  # PS_DR1
]

delta_bic = bic - bic_values_for_M[data_choice]

print("Delta BIC:", delta_bic)


max_iters = 6000
ndim = 4
nwalkers = 22

p0 = []
for i in range(nwalkers):
    M_init = np.random.uniform(0.3, 0.6)
    D_init = np.random.uniform(0, 0.2)
    d_ra_init = np.random.uniform(0, 2 * np.pi)
    d_dec_init = np.random.uniform(-np.pi / 2, np.pi / 2)
    p0.append([M_init, D_init, d_ra_init, d_dec_init])

print("Running MCMC with", cpu_count(), "cores for", max_iters, "iterations...")

with Pool() as pool:
    filename = "chain_" + name + "_MD.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    sampler.run_mcmc(p0, max_iters, progress=True)

print("Completed. Mean acceptance fraction =", round(np.mean(sampler.acceptance_fraction), 3))


reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain(discard=500, flat=True)  # manually discard burn-in if desired
log_prob_samples = reader.get_log_prob(discard=500, flat=False)
log_prob_samples_flat = reader.get_log_prob(discard=500, flat=True)

# Filtering high likelihood samples
lnlike = -1.e100
mask_like = log_prob_samples_flat > lnlike
samples = samples[mask_like, :]

log_prob_samples_flat = log_prob_samples_flat[mask_like]

burnin=500




nsteps = len(log_prob_samples[:,0])
ndim = len(samples[0,:])
nwalkers = len(log_prob_samples[0,:])
nsteps_mask = int(len(samples[:,0])/nwalkers)

print("nsteps:", nsteps, ", nsteps_mask:", nsteps_mask, ", ndim:", ndim, ", nwalkers:", nwalkers)

x = list(range(1, nsteps + 1))
for i in range(nwalkers):
    like_i = log_prob_samples[:,i]              
    mask = like_i > lnlike

    indices = np.where(mask)[0]

    plt.scatter(np.array(x)[indices] + burnin, like_i[indices], s=10)

plt.xlabel("Walker step")

plt.ylabel("ln(Likelihood)")

plt.savefig("Plots/"+name+"_MD_walkers.png", bbox_inches="tight")



labels = [r'${\rm \widehat{M}}$', r'${\rm \widehat{D}}$', r'${\rm d_{\alpha}}$', r'${\rm d_{\delta}}$', r'${\rm \widehat{Q}}$', r'${\rm q_{1,\alpha}}$', r'${\rm q_{1,\delta}}$', r'${\rm q_{2,\alpha}}$', r'${\rm q_{2,\delta}}$']



plt.clf()

truth_values = [None] * ndim
truth_values[labels.index(r'${\rm \widehat{M}}$')] = 0.5

corner.corner(samples, labels=labels, show_titles=True, quantiles=[], plot_datapoints=False,
              title_kwargs={"fontsize": 18}, title_fmt='.3f', label_kwargs={"fontsize": 18},
              verbose=True, truths=truth_values)

plt.savefig("Plots/"+name+"_MD_corner.png", bbox_inches="tight")

param_means = np.mean(samples, axis=0)
param_std = np.std(samples, axis=0)

param_quotes = []

for i, (mean, std) in enumerate(zip(param_means, param_std)):
    if mean - 2 * std < 0:
        p68 = np.percentile(samples[:, i], 68)
        p95 = np.percentile(samples[:, i], 95)
        param_quotes.append(f'<{p68:.3f} at 1 sigma, <{p95:.3f} at 2 sigma')
    else:
        param_quotes.append("use value given by corner plot")

for i, param_quote in enumerate(param_quotes):
    print(f'Parameter {i+1}: {param_quote}')
