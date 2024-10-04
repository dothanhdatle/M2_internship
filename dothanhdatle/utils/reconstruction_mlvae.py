import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error


### Reconstruction using group
def reconstruct_one(model, nirs_data, tar_variete, tar_env, device):
    """
    Function for reconstruction the missing NIRS by groups
    
    Parameters:
    model: model Disentangled VAE used to reconstruct
    tar_variete: target variete
    tar_env: the target environment
    device: cpu or gpu for training
    """

    model.eval()

    column_names = nirs_data.columns.tolist()

    with torch.no_grad():
        # all nirs of the target variete
        var_data = nirs_data[nirs_data.GenoID == tar_variete]
        var_label = list(var_data.GenoID)
        variete_nirs = np.array(var_data.iloc[:,2:])
        variete_nirs = torch.from_numpy(variete_nirs).to(device).float()
        variete_nirs = variete_nirs.reshape(variete_nirs.shape[0],1,-1)

        # Encode latent vector for the target variete
        variete_mu, variete_logvar, _, _ = model.encode(variete_nirs)
        grouped_mu_var, grouped_logvar_var = model.accumulate_group_evidence(variete_mu, variete_logvar, var_label, device)
        z_variete = model.group_wise_reparameterize(grouped_mu_var, grouped_logvar_var, var_label, train=False)

        # all nirs for the missing environment
        env_data = nirs_data[nirs_data.Environnement == tar_env]
        env_label = list(env_data.Environnement)
        env_nirs = np.array(env_data.iloc[:,2:])
        env_nirs = torch.from_numpy(env_nirs).to(device).float()
        env_nirs = env_nirs.reshape(env_nirs.shape[0],1,-1)                 

        # Encode latent vector for the target environment
        _, _, env_mu, env_logvar = model.encode(env_nirs)
        grouped_mu_env,  grouped_logvar_env = model.accumulate_group_evidence(env_mu, env_logvar, env_label, device)
        z_env = model.group_wise_reparameterize(grouped_mu_env, grouped_logvar_env, env_label, train=False)

        # Concatenate to get new latent variable for the target nirs
        new_z = torch.cat((z_variete[0], z_env[0]), dim=-1).reshape(1,-1).to(device)

        # Decode
        model.pxz_distribution.param = model.decode(new_z)
        new_nirs = model.pxz_distribution.mean() # new nirs generated
        #new_nirs = new_nirs.detach().numpy()
        new_nirs = new_nirs.reshape(-1)

    return new_nirs

def reconstruct_all_mlvae(model, nirs_data, device):
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
            rec_tar_nirs = reconstruct_one(model, nirs_data,
                    tar_variete, tar_env, device).detach().cpu().numpy()
            
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