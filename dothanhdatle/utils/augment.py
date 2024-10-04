import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from scipy.signal import savgol_filter

# Data Augmentation
def add_gaussian_noise(spectra, noise_level=0.01, random_state = 42):
    np.random.RandomState(random_state)
    noise = np.random.normal(0, 1, spectra.shape)
    return spectra + noise_level*noise

def wavelength_shift(spectra, shift):
    return np.roll(spectra, shift)

def amplitude_scaling(spectra, scale_factor=1.2):
    return spectra * scale_factor

def savitzky_golay_filter(spectra, window_length=5, polyorder=2):
    return savgol_filter(spectra, window_length, polyorder)