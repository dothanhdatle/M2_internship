import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self, duplicate_replace = True, derivative = False, normalize = False, normalize_type = 'standard'):
        self.duplicate_replace = duplicate_replace
        self.derivative = derivative
        self.normalize = normalize
        self.normalize_type = normalize_type

    def preprocess(self, data):
        if self.duplicate_replace:
            data = data.groupby(['GenoID', 'Environnement']).mean().reset_index()
        if self.derivative:
            data = derivative_data(data)
        if self.normalize:
            if self.normalize_type=='standard':
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
            elif self.normalize_type=='min-max':
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)
            elif self.normalize_type=='snv':
                data = SNV(data) 
        return data
        
## Derivatives
def derivative_one_nirs(wave_lengths, spectra):
    derivative = np.diff(spectra, axis = 1)/np.diff(wave_lengths)
    w = 0.5*(np.array(wave_lengths[1:])+np.array(wave_lengths[:-1]))

    return w, derivative

def derivative_data(nirs_data):
    # Extract wavelengths
    wave_lengths = nirs_data.columns[2:]
    wave_lengths = [float(s.replace(',', '.')) for s in wave_lengths]
    
    derivative_data = []

    # compute derivative for each spectra
    for index, row in nirs_data.iterrows():
        spectra = row.iloc[2:].values.astype(float).reshape(1,-1)
        w, derivative = derivative_one_nirs(wave_lengths, spectra)
        derivative_data.append(derivative.flatten())
    
    # Create new derivatives data
    derivative_data = pd.DataFrame(derivative_data, columns=w)
    
    # Add GenoID and Environnement columns to the derivative DataFrame
    derivative_data.insert(0, 'GenoID', nirs_data['GenoID'].values)
    derivative_data.insert(1, 'Environnement', nirs_data['Environnement'].values)
    
    return derivative_data

## Normalization
def SNV(nirs_data):
    """
    Standard Normal Variate to normalize spectra data
    """
    snv_df = nirs_data.copy()
    mean = nirs_data.iloc[:,2:].mean(axis = 1)
    std = nirs_data.iloc[:,2:].std(axis = 1)
    mean_centered = nirs_data.iloc[:,2:].sub(nirs_data.iloc[:,2:].mean(axis = 1), axis = 0)
    snv_data = mean_centered.div(nirs_data.iloc[:,2:].std(axis = 1), axis = 0)
    snv_df.iloc[:,2:] = snv_data
    return snv_df, mean, std

def reverse_SNV(snv_data, mean, std):
    """
    Reverse the Standard Normal Variate to get raw spectra data
    """
    raw_df = snv_data.copy()
    mean_centered =  raw_df.iloc[:,2:].mul(std, axis = 0)
    raw_data = mean_centered.add(mean, axis = 0)
    raw_df.iloc[:,2:] = raw_data
    return raw_df