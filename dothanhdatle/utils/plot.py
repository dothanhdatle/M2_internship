import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations


# Plotting
def plot_nirs_random(data, nb_nirs = 10, fig_size = (10,6), line_style = '-', new_fig = True):
    """
    Plots some random NIR spectra in the data

    Parameters:
    data: NIRS data.
    nb_nirs: number of NIR spectra
    """

    samples = data.sample(n = nb_nirs)

    wavelengths = data.iloc[:,2:].columns
    wavelengths = [float(s.replace(',', '.')) for s in wavelengths]
    
    if new_fig:
        plt.figure(figsize=fig_size)
        min_wavelength = np.array(wavelengths).min()
        max_wavelength = np.array(wavelengths).max()
        tick_positions = list(range(int(min_wavelength), int(max_wavelength) + 1, 200))

    variete_list = list(samples.GenoID)
    environment_list = list(samples.Environnement)
    nirs = samples.iloc[:,2:].values

    # Plot the spectra
    for i in range(len(nirs)):
        plt.plot(wavelengths, nirs[i], linestyle=line_style, label=f'{variete_list[i]}, {environment_list[i]}')

    if new_fig:
        plt.xticks(tick_positions, tick_positions)
        plt.xlabel('Wavelengths')
        plt.title(f'{nb_nirs} random NIR Spectra')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_nirs_variete(data, variete, environment_list = [], fig_size = (10,6), line_style = '-', new_fig = True):
    """
    Plots the NIR spectra for a given variete in all available environment.

    Parameters:
    data: NIRS data.
    variete: the variete (GenoID) to plot.
    """
    wavelengths = data.iloc[:,2:].columns
    wavelengths = [float(s.replace(',', '.')) for s in wavelengths]

    if new_fig:
        plt.figure(figsize=fig_size)
        min_wavelength = np.array(wavelengths).min()
        max_wavelength = np.array(wavelengths).max()
        tick_positions = list(range(int(min_wavelength), int(max_wavelength) + 1, 200))

    if environment_list == []:
        filtered = data[data.GenoID == variete]
        if filtered.empty:
            print(f'No data found for the variete: {variete}')
            return
        environment_list = list(filtered.Environnement)
        nirs = filtered.iloc[:, 2:].values

        # Plot the spectra
        for i in range(len(nirs)):
            plt.plot(wavelengths, nirs[i], linestyle = line_style, label=f'{variete}, {environment_list[i]}')
    else:
        for environment in environment_list:
            # Filter the data for the specific GenoID and environment
            filtered = data[(data.GenoID == variete) & (data.Environnement == environment)]
        
            if filtered.empty:
                print(f'No data found for the environment: {environment}')
                continue
            
            nirs = filtered.iloc[:, 2:].values
            # Plot the spectra
            for i in range(len(nirs)):
                plt.plot(wavelengths, nirs[i], linestyle = line_style, label=f'{environment}')
    
    if new_fig:
        plt.xticks(tick_positions, tick_positions)
        plt.xlabel('Wavelengths')
        plt.title(f'NIR Spectra for variete {variete}')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_nirs_environment(data, environment, variete_list = [], fig_size = (10,6), line_style = '-', new_fig = True):
    """
    Plots the NIR spectra for a given environment and different varietes.

    Parameters:
    data: NIRS data.
    variete: the variete (GenoID) to plot.
    """
    wavelengths = data.iloc[:,2:].columns
    wavelengths = [float(s.replace(',', '.')) for s in wavelengths]

    if new_fig:
        plt.figure(figsize=fig_size)
        min_wavelength = np.array(wavelengths).min()
        max_wavelength = np.array(wavelengths).max()
        tick_positions = list(range(int(min_wavelength), int(max_wavelength) + 1, 200))

    if variete_list == []:
        filtered = data[data.Environnement == environment]
        if filtered.empty:
            print(f'No data found for the environment: {environment}')
            return
        variete_list = list(filtered.GenoID)
        nirs = filtered.iloc[:, 2:].values

        # Plot the spectra
        for i in range(len(nirs)):
            plt.plot(wavelengths, nirs[i], linestyle = line_style, label=f'{variete_list[i]}, {environment}')
    else:
        for variete in variete_list:
            # Filter the data for the specific GenoID and environment
            filtered = data[(data.GenoID == variete) & (data.Environnement == environment)]
        
            if filtered.empty:
                print(f'No data found for the variete: {variete}')
                continue
            
            nirs = filtered.iloc[:, 2:].values
            # Plot the spectra
            for i in range(len(nirs)):
                plt.plot(wavelengths, nirs[i], linestyle = line_style, label=f'{variete}')
    
    if new_fig:
        plt.xticks(tick_positions, tick_positions)
        plt.xlabel('Wavelengths')
        plt.title(f'NIR Spectra for environment {environment}')
        plt.legend()
        plt.grid(True)
        plt.show()

def nirs_comparison_plot(ground_truth, recon_data, variete, environment, fig_size = (10,6), new_fig = True):
    """
    Plots the NIR spectra for a given environment and different varietes.

    Parameters:
    recon_data: the reconstructed missing NIRS data.
    ground_truth: the ground truth NIRS data.
    variete: The variete (GenoID) to plot.
    environment: The environment to plot.
    """

    # Filter the data for the specific variete and environment in reconstructed data
    rec_filtered = recon_data[(recon_data.GenoID == variete) & (recon_data.Environnement == environment)]
    
    # Filter the data for the specific GenoID and environment in ground truth data
    gt_filtered = ground_truth[(ground_truth.GenoID == variete) & (ground_truth.Environnement == environment)]
    
    if rec_filtered.empty:
        print(f"No reconstructed data found for variete: {variete} in environment: {environment}")
        return
    
    if gt_filtered.empty:
        print(f"No ground truth data found for variete: {variete} in environment: {environment}")
        return
    
    wavelengths = recon_data.iloc[:,2:].columns
    wavelengths = [float(s.replace(',', '.')) for s in wavelengths]
    
    rec_nirs = rec_filtered.iloc[:, 2:].values
    gt_nirs = gt_filtered.iloc[:, 2:].values

    if new_fig:
        plt.figure(figsize=fig_size)
        min_wavelength = np.array(wavelengths).min()
        max_wavelength = np.array(wavelengths).max()
        tick_positions = list(range(int(min_wavelength), int(max_wavelength) + 1, 200))

    for i in range(len(rec_nirs)):
        plt.plot(wavelengths, rec_nirs[i], linestyle='--', label=f'Generative: {variete}, {environment}', alpha=0.6)
        
    for i in range(len(gt_nirs)):
        plt.plot(wavelengths, gt_nirs[i], linestyle='-', label=f'Ground Truth: {variete}, {environment}', alpha=0.6)

    if new_fig:
        plt.xticks(tick_positions, tick_positions)
        plt.xlabel('Wavelengths')
        plt.title(f'Reconstructed vs. Ground truth NIRS')
        plt.legend()
        plt.grid(True)
        plt.show()