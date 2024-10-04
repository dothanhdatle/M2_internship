import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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