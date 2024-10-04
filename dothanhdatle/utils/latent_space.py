import numpy as np
import pandas as pd
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


### Latent space visualization
def latent_cvae(model, data, device):
    model.eval()
    geno_id = data.geno_id
    env_label = data.env_label
    nirs = data.spec
    nirs = torch.from_numpy(nirs).to(device).float()
    nirs = nirs.reshape(nirs.shape[0],1,-1)
    z_list = []
    with torch.no_grad():
        for var, env, nir in zip(geno_id, env_label, nirs):
            var_code, env_code = data.get_code(var, env)
            x_cond = torch.cat((nir, var_code.unsqueeze(0).to(device), env_code.unsqueeze(0).to(device)), dim = 1)
            mu, logvar = model.encode(x_cond)
            z = model.reparameterize(mu, logvar, train = False)
            z_list.append(z.squeeze().cpu().numpy())

    return np.array(z_list), geno_id, env_label

def latent_dvae(model, data, device):
    model.eval()
    var_label, env_label = [],[]
    latent_list = []
    with torch.no_grad():
        for i, (x1,x2,var,env1,env2) in enumerate(data):
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()

            variete_mu1, variete_logvar1, env_mu1, env_logvar1 = model.encode(x1)
            z_env1 = model.reparameterize(env_mu1, env_logvar1, train = False)

            variete_mu2, variete_logvar2, env_mu2, env_logvar2 = model.encode(x2)
            z_env2 = model.reparameterize(env_mu2, env_logvar2, train = False)

            # Average the latent space for variete
            variete_mu = (variete_mu1 + variete_mu2)/2
            variete_logvar = torch.log((variete_logvar1.exp()*variete_logvar2.exp())**0.5)
            z_variete = model.reparameterize(variete_mu, variete_logvar, train = False)

            # Latent vectors
            z1 = torch.cat((z_variete, z_env1), dim=-1)
            z2 = torch.cat((z_variete, z_env2), dim=-1)

            z1 = z1.squeeze().cpu().numpy()
            z2 = z2.squeeze().cpu().numpy()

            latent_list.extend([z1,z2])
            var_label.extend([var]*2)
            env_label.extend([env1,env2])
    
    return np.array(latent_list), np.array(var_label), np.array(env_label)

def latent_mlvae(model, data, device):
    model.eval()
    geno_id = data.geno_id
    environments = data.env_label
    df = data.spec
    var_set = set(geno_id)
    var_labels, env_labels = [],[]
    latent_list = []

    with torch.no_grad():
        for v in var_set:
            indices = [idx for idx, val in enumerate(geno_id) if val == v]
            vars = geno_id[indices]
            envs = environments[indices]
            nirs = df[indices]
            nirs = torch.from_numpy(nirs).to(device).float()
            nirs = nirs.reshape(nirs.shape[0],1,-1)
            variete_mu, variete_logvar, env_mu, env_logvar = model.encode(nirs)
            grouped_mu, grouped_logvar = model.accumulate_group_evidence(variete_mu, variete_logvar, vars, device)
            z_env = model.reparameterize(env_mu, env_logvar, train=False)
            z_variete = model.group_wise_reparameterize(grouped_mu, grouped_logvar, vars, train=False)
            z_variete = z_variete.to(device)
            z_env = z_env.to(device)
            z = torch.cat((z_variete, z_env), dim=-1)
            z = z.cpu().numpy()
            latent_list.extend(list(z))
            var_labels.extend(list(vars))
            env_labels.extend(list(envs))
    
    return np.array(latent_list), np.array(var_labels), np.array(env_labels)

# Dimension Reduction

def dimension_reduce(data, n_components = 2, nb_neighbors = 15, min_dist = 0.1, perplexity=30, type = 'pca'):
    if type == 'pca':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
    elif type == 'tsne':
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        reduced_data = tsne.fit_transform(data)
    elif type == 'umap':
        umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=nb_neighbors, min_dist=min_dist)
        reduced_data = umap_reducer.fit_transform(data)

    return reduced_data

# plot
def plot_latent_space(latent_vec, labels, title):
    plt.figure(figsize=(10, 8))
    fig = sns.scatterplot(x=latent_vec[:, 0], y=latent_vec[:, 1], hue=labels)
    fig.set(xlabel='Dimension 1', ylabel='Dimension 2', title = title)
    plt.show()