import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import random

### Reconstruction using group
def reconstruct_one(model, var, env, device):
    """
    Function for reconstruction the missing NIRS by groups
    
    Parameters:
    model: model Disentangled VAE used to reconstruct
    tar_variete: target variete
    tar_env: the target environment
    device: cpu or gpu for training
    """

    model.eval()
    with torch.no_grad():
        # generate random noise
        torch.manual_seed(42)
        random.seed(42)
        z = torch.randn(1, model.latent_dim).to(device)
        z.reshape(1, -1)
        var = var.unsqueeze(0).to(device)
        env = env.unsqueeze(0).to(device)
        # concat with geno and env
        z_cond = torch.cat((z, var, env), dim = 1)
        model.pxz_distribution.param = model.decode(z_cond)
        new_nirs = model.pxz_distribution.mean()
        new_nirs = new_nirs.reshape(-1)
    
    return new_nirs
    

def reconstruct_all_cvae(model, train_loader, nirs_data, device):
    """
    Function for reconstruction all missing NIRS

    Parameters:
    model: model used to reconstruct
    nirs_data: nirs data frame
    device: cpu or gpu for training 
    """
    column_names = nirs_data.columns.tolist()
    #column_names.remove('Supplement')
    recon_miss_data = []

    variete_list = nirs_data.GenoID.unique()  # list of variete
    env_list = nirs_data.Environnement.unique()  # list of Environment

    for tar_variete in variete_list:
        # list of environment that having nirs for the target variete
        env_have = nirs_data[nirs_data.GenoID == tar_variete].Environnement.unique()

        # list of environments that missing nirs for the target variete
        env_missing = [env for env in env_list if env not in env_have]
        
        for tar_env in env_missing:
            # Reconstruction nirs of (tar_variete, env) from (src_variete, env)
            tar_var_code, tar_env_code = train_loader.dataset.get_code(tar_variete, tar_env)
            rec_tar_nirs = reconstruct_one(model, 
                                           var = tar_var_code, 
                                           env = tar_env_code, 
                                           device = device).detach().cpu().numpy()
            
            # new row nirs
            new_nirs = [tar_variete, tar_env] + list(rec_tar_nirs)

            # Add new generated NIRS to the list
            recon_miss_data.append(new_nirs)

    # Convert list of rows to DataFrame 
    recon_miss = pd.DataFrame(recon_miss_data, columns=column_names)
    return recon_miss

def baseline_prediction(nirs_data, env):
    nirs_env = nirs_data[nirs_data.Environnement == env].iloc[:,2:]
    return np.array(nirs_env.mean(axis = 0))

def baseline_prediction_data(df_gt):
    column_names = df_gt.columns.tolist()
    recon_miss_data = []

    variete_list = df_gt.GenoID.unique()  # list of variete
    env_list = df_gt.Environnement.unique()  # list of Environment

    for tar_variete in variete_list:
        # list of environment that having nirs for the target variete
        env_have = df_gt[df_gt.GenoID == tar_variete].Environnement.unique()

        # list of environments that missing nirs for the target variete
        env_missing = [env for env in env_list if env not in env_have]

        for tar_env in env_missing:
            rec_tar_nirs = baseline_prediction(df_gt, tar_env)
            
            # new row nirs
            new_nirs = [tar_variete, tar_env] + list(rec_tar_nirs)

            # Add new generated NIRS to the list
            recon_miss_data.append(new_nirs)

    # Convert list of rows to DataFrame 
    recon_miss = pd.DataFrame(recon_miss_data, columns=column_names)
    return recon_miss

def mse_miss_recon(miss_recon, ground_truth):
    mse_loss = 0
    mse_list = []
    num_rec = 0
    mse_df = miss_recon.copy()
    for i in range(len(miss_recon)):
        variete = miss_recon.iloc[i,0]
        env = miss_recon.iloc[i,1]
        nirs_rec = miss_recon.iloc[i,2:].values
        nirs_gt = ground_truth[(ground_truth.GenoID == variete) & (ground_truth.Environnement == env)].iloc[:,2:].values
        if len(nirs_gt) == 0:
            print(f'No NIRS groundtruth available for the {variete} in the environment {env}')
            index = mse_df[(mse_df['GenoID'] == variete) & (mse_df['Environnement'] == env)].index
            mse_df.drop(index , inplace=True)
            continue
        else:
            nirs_gt = nirs_gt[0]
        
        mse = mean_squared_error(nirs_gt, nirs_rec)
        mse_list.append(mse)
        mse_loss += mse
        num_rec += 1
    
    avg_mse = mse_loss/num_rec
    mse_df['MSE'] = mse_list
    
    return avg_mse, mse_df