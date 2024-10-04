import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

class NIRS_Dataset_CVAE(Dataset):
    def __init__(self, file_path = None, data = None, one_hot_encode=False):
        if file_path != None:
            self.file_path = file_path
            self.nirs = pd.read_csv(self.file_path)
        else:
            self.nirs = data
        
        self.spec = np.array(self.nirs.iloc[:,2:])

        self.geno_id = np.array(self.nirs.GenoID)
        self.env_label = np.array(self.nirs.Environnement)

        self.one_hot_encode = one_hot_encode
        if self.one_hot_encode:
            # One-Hot Encoder for variety and environment
            self.var_encoder = OneHotEncoder(sparse_output=False)
            self.env_encoder = OneHotEncoder(sparse_output=False)
                
            self.var_encoder.fit(self.nirs.iloc[:, 0].values.reshape(-1, 1))
            self.env_encoder.fit(self.nirs.iloc[:, 1].values.reshape(-1, 1))

    def __getitem__(self, index):
        nir = self.nirs.iloc[index, 2:].values.astype(float)
        var = self.nirs.iloc[index, 0]
        env = self.nirs.iloc[index, 1]

        if self.one_hot_encode:
            var_code = self.var_encoder.transform([[var]])[0]
            env_code = self.env_encoder.transform([[env]])[0]
            return torch.from_numpy(nir).unsqueeze(0).float(), torch.tensor(var_code, dtype=torch.float32), \
                    torch.tensor(env_code, dtype=torch.float32), var, env
        else:
            return torch.from_numpy(nir).unsqueeze(0).float(), var, env
        
    def __len__(self):
        return len(self.nirs)
    
    def get_code(self, var, env):
        var_code = self.var_encoder.transform([[var]])[0]
        env_code = self.env_encoder.transform([[env]])[0]
        return torch.tensor(var_code, dtype=torch.float32), torch.tensor(env_code, dtype=torch.float32)
    
class NIRS_Dataset_DVAE(Dataset):
    def __init__(self, file_path = None, data = None):
        if file_path != None:
            self.file_path = file_path
            self.nirs = pd.read_csv(self.file_path)
        else:
            self.nirs = data

        self.geno_ids = self.nirs.GenoID.unique()  # array of unique full GenoID
        self.geno_nir = {}
        self.geno_env = {}
        self.nir_pairs = []
        self.nir_pairs_ids = []
        self.env_pairs = []

        for geno in self.geno_ids:
            geno_data = self.nirs[self.nirs.GenoID == geno]  
            geno_nir = geno_data.iloc[:, 2:].values  
            env = geno_data['Environnement'].values
            
            combined = np.column_stack((geno_nir, env))
            
            # Shuffle
            np.random.shuffle(combined)

            geno_nir = combined[:, :-1] 
            env = combined[:, -1] 
    
            self.geno_nir[geno] = geno_nir
            self.geno_env[geno] = env
            
            if geno_nir.shape[0] != 1:
                half_size = geno_nir.shape[0] // 2
                geno_nir1 = geno_nir[:half_size]
                geno_nir2 = geno_nir[half_size:half_size + half_size]
                geno_nir_pairs = np.array(list(zip(geno_nir1, geno_nir2)))  

                # Pair up the environments in the same way as NIR data
                env1 = env[:half_size]
                env2 = env[half_size:half_size + half_size]
                env_pairs = np.array(list(zip(env1, env2)))
            else:
                geno_nir_pairs = np.array([(geno_nir[0], geno_nir[0])])
                env_pairs = np.array([(env[0], env[0])])
                
            self.nir_pairs.append(geno_nir_pairs)
            self.nir_pairs_ids.extend([geno] * len(geno_nir_pairs))
            self.env_pairs.append(env_pairs)

        self.nir_pairs = np.vstack(self.nir_pairs)  # Stack all pairs together
        self.nir_pairs_ids = np.array(self.nir_pairs_ids)  # Convert ids to numpy array for faster indexing 
        self.env_pairs = np.vstack(self.env_pairs)
    
    def shuffle_data(self):
        self.nir_pairs = []
        self.nir_pairs_ids = []
        self.env_pairs = []
        for geno in self.geno_ids:
            combined = np.column_stack((self.geno_nir[geno], self.geno_env[geno]))
            np.random.shuffle(combined)

            nir = combined[:, :-1] 
            env = combined[:, -1]

            if nir.shape[0] != 1:
                half_size = nir.shape[0] // 2
                nir1 = nir[:half_size]
                nir2 = nir[half_size:half_size + half_size]
                nir_pairs = np.array(list(zip(nir1, nir2)))

                # Pair environments similarly
                env1 = env[:half_size]
                env2 = env[half_size:half_size + half_size]
                env_pairs = np.array(list(zip(env1, env2)))
            else:
                nir_pairs = np.array([(nir[0], nir[0])])
                env_pairs = np.array([(env[0], env[0])])
                
            self.nir_pairs.append(nir_pairs)
            self.nir_pairs_ids.extend([geno] * len(nir_pairs))
            self.env_pairs.append(env_pairs)

        self.nir_pairs = np.vstack(self.nir_pairs)
        self.nir_pairs_ids = np.array(self.nir_pairs_ids) 
        self.env_pairs = np.vstack(self.env_pairs)
    
    def __getitem__(self, index):
        nir1, nir2 = self.nir_pairs[index]
        nir1 = nir1.astype(float)
        nir2 = nir2.astype(float)
        env1, env2 = self.env_pairs[index]
        return torch.from_numpy(nir1).unsqueeze(0).float(), torch.from_numpy(nir2).unsqueeze(0).float(), \
            self.nir_pairs_ids[index], env1, env2

    def __len__(self):
        return len(self.nir_pairs)
    
    def get_nir_env(self, GenoID, Environment):
        nir = np.array(self.nirs[(self.nirs.GenoID == GenoID) & (self.nirs.Environnement == Environment)].iloc[:,2:]).values
        return nir

    def get_all_nir(self, GenoID):
        return self.geno_nir[GenoID]
    

class NIRS_Dataset(Dataset):
    def __init__(self, file_path = None, data = None):
        if file_path != None:
            self.file_path = file_path
            self.nirs = pd.read_csv(self.file_path)
        else:
            self.nirs = data
        
        self.spec = np.array(self.nirs.iloc[:,2:])

        self.geno_id = np.array(self.nirs.GenoID)
        self.env_label = np.array(self.nirs.Environnement)

    def __getitem__(self, index):
        nir = self.nirs.iloc[index, 2:].values.astype(float)
        var = self.nirs.iloc[index, 0]
        env = self.nirs.iloc[index, 1]
        
        return torch.from_numpy(nir).unsqueeze(0).float(), var, env
        
    def __len__(self):
        return len(self.nirs)
    
    def get_num_var(self, env):
        env_df = self.nirs[self.nirs.Environnement == env]
        return env_df.GenoID.shape[0]
    
    def get_num_env(self, var):
        var_df = self.nirs[self.nirs.GenoID == var]
        return var_df.Environnement.shape[0]
    
# Mini-batch sampling over variety
class Batch_Sampler:
    def __init__(self, data, num_vars):
        self.data = data
        self.num_vars = num_vars
        self.var_groups = self.group_by_var()

    def group_by_var(self):
        groups = defaultdict(list)
        for i, var in enumerate(self.data.geno_id):
            groups[var].append(i)
        # Convert to a list of lists
        grouped_vars = list(groups.values())
        # Shuffle the list to ensure random sampling
        np.random.shuffle(grouped_vars)        
        return grouped_vars

    def __iter__(self):
        batch_ind = []
        current_vars = set()
        for indices in self.var_groups:
            var = self.data.geno_id[indices[0]]
            batch_ind.append(indices)
            current_vars.add(var)
            if len(current_vars) == self.num_vars:
                batch = [item for sublist in batch_ind for item in sublist]
                yield batch
                batch_ind = []
                current_vars = set()
        # Yield the remaining batch if it exists
        if batch_ind:
            batch = [item for sublist in batch_ind for item in sublist]
            yield batch

    def __len__(self):
        return len(self.var_groups) // self.num_vars

# collate function
def collate_func(batch):
    nir = batch[0][0].reshape(batch[0][0].shape[1],1,-1)
    var = tuple(batch[0][1])
    env = tuple(batch[0][2])
    return nir, var, env


def split_data(df, geno_column='GenoID', test_size=0.2, random_state=42):
    """
    Split the spectral dataframe into training and test sets with an 80:20 ratio,
    ensuring that all spectra for each GenoID are kept together.

    Parameters:
    df (pd.DataFrame): The input DataFrame with spectral data.
    geno_column (str): The column name for GenoID (default is 'GenoID').
    env_column (str): The column name for Environnement (default is 'Environnement').
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2 for 20%).
    random_state (int): The random seed for reproducibility (default is 42).

    Returns:
    pd.DataFrame, pd.DataFrame: The training and test DataFrames.
    """
    # Get the unique GenoIDs
    unique_geno_ids = df[geno_column].unique()

    # Split the GenoIDs into training and test sets
    geno_train, geno_test = train_test_split(unique_geno_ids, test_size=test_size, random_state=random_state)

    # Create training and test sets by filtering the original DataFrame
    train_df = df[df[geno_column].isin(geno_train)]
    test_df = df[df[geno_column].isin(geno_test)]

    return train_df, test_df

def remove_data(df, percentage):
    # total number of rows to remove
    total_rows = len(df)
    rows_to_remove = int(total_rows * percentage)
    
    geno_groups = df.groupby('GenoID')
    env_groups = df.groupby('Environnement')

    # Each GenoID has at least one spectrum left
    geno_keep_indices = geno_groups.apply(lambda x: x.sample(1, random_state=42)).index.get_level_values(1)
    
    # Each Environnement has at least one spectrum left
    env_keep_indices = env_groups.apply(lambda x: x.sample(1, random_state=42)).index.get_level_values(1)
    
    # Indices to be kept
    indices_keep = pd.Index(geno_keep_indices).union(env_keep_indices)
    
    # indice could be remove
    removable_indices = df.index.difference(indices_keep)
    
    # Randomly select indices to remove
    rows_remove = np.random.choice(removable_indices, size=rows_to_remove, replace=False)
    
    # Drop the selected rows
    df = df.drop(rows_remove)
    
    return df


        
    

