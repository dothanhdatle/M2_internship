import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
#from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torch.optim as optim

from utils.append_dict import append_elbo_dict
from utils.reconstruction_dvae import reconstruct_one, reconstruct_all_dvae, mse_miss_recon
class LinearInit(nn.Module):
    def __init__(self, in_dim, out_dim, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim)

        # initialize weights
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
 
    def forward(self, x):
        return self.linear_layer(x)
 
class DVAE(nn.Module):
    """
    Disentangled Variational AutoEncoder:
    pxz_distribution:
    input_dim: input size
    hidden_dims: list of hidden layer size
    latent_dim: latent size = size of latent vector for variete + size of latent vector for environment
    vari_latent_size: size of latent vector for variete
    """

    def __init__(self, pxz_distribution, input_dim, hidden_dims,
                  latent_dim, vari_latent_size, beta = 1, beta_annealing = False, anneal_type = 'cycle_linear'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vari_latent_size = vari_latent_size
        self.pxz_distribution = pxz_distribution(input_dim)
        self.beta = beta
        self.beta_annealing = beta_annealing
        if beta_annealing:
            self.anneal_type = anneal_type
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]

        # Build Encoder

        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(
                nn.Sequential(
                    LinearInit(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
            )

        self.enc_mlp = nn.Sequential(*modules)
        self.enc_linear = LinearInit(hidden_dims[-1], 2*latent_dim)
        self.variete = LinearInit(2*latent_dim, self.vari_latent_size*2)
        self.env = LinearInit(2*latent_dim, (latent_dim - self.vari_latent_size)*2)

        # Build decoder
        modules = []

        for in_dim, out_dim in zip([latent_dim] + hidden_dims[::-1], hidden_dims[::-1]):
            modules.append(
                nn.Sequential(
                    LinearInit(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
            )
        
        self.dec_mlp = nn.Sequential(*modules)
        # last linear layer to get decoder output
        self.dec_linear = LinearInit(hidden_dims[::-1][-1], self.pxz_distribution.decoder_output_dim) 

    def encode(self, x):
        """
        Encodes the input spectra x by passing through the encoder network
        and returns the mean and log variance of latent variables for variete and environment.
        x: input data
        ouput: mean (mu) and log variance (logvar) of latent variable z
        """

        shape = x.shape
    
        output = self.enc_mlp(x)
        output = output.reshape(shape[0], -1)
        output = self.enc_linear(output)
        variete = self.variete(output)
        env = self.env(output)

        variete_mu = variete[:,:self.vari_latent_size]
        variete_logvar = variete[:,self.vari_latent_size:]
        env_mu = env[:,:(self.latent_dim-self.vari_latent_size)]
        env_logvar = env[:,(self.latent_dim-self.vari_latent_size):]
        
        return variete_mu, variete_logvar, env_mu, env_logvar
    
    def decode(self, z):
        """
        Decodes the latent variable z by passing through the decoder network
        z: latent variable z
        """

        output = self.dec_mlp(z)
        output = output.view(z.shape[0],-1, self.hidden_dims[::-1][-1])
        output = self.dec_linear(output)
        return self.pxz_distribution(output)
    
    def reparameterize(self, mu, logvar, train = True):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        mu: Mean of the latent Gaussian variable z
        logvar: log variance of the latent Gaussian variable z
        output: sample latent variable z
        """
        if train:
            std = torch.exp(torch.mul(logvar, 0.5))
            #torch.manual_seed(42)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
    
    def forward(self, x1, x2, train=True):
        # Encode nirs pair
        variete_mu1, variete_logvar1, env_mu1, env_logvar1 = self.encode(x1)
        z_env1 = self.reparameterize(env_mu1, env_logvar1, train = train)

        variete_mu2, variete_logvar2, env_mu2, env_logvar2 = self.encode(x2)
        z_env2 = self.reparameterize(env_mu2, env_logvar2, train = train)

        # Average the latent space for variete
        variete_mu = (variete_mu1 + variete_mu2)/2
        #variete_logvar = (variete_logvar1+variete_logvar2)/2
        variete_logvar = torch.log((variete_logvar1.exp()*variete_logvar2.exp())**0.5)
        z_variete = self.reparameterize(variete_mu, variete_logvar, train = train)

        # Latent vectors
        self.z1 = torch.cat((z_variete, z_env1), dim=-1)
        self.z2 = torch.cat((z_variete, z_env2), dim=-1)

        ## parameters of distribution of sample 1
        q_z1_mu = torch.cat((variete_mu, env_mu1), dim=-1)
        q_z1_logvar = torch.cat((variete_logvar, env_logvar1), dim=-1)

        ## parameters of distribution of sample 2
        q_z2_mu = torch.cat((variete_mu, env_mu2), dim=-1)
        q_z2_logvar = torch.cat((variete_logvar, env_logvar2), dim=-1)

        # Reconstruction
        self.pxz_distribution.param = self.decode(self.z1)
        rec_term_x1 = torch.mean(self.pxz_distribution.llkh(x1))
        rec_x1 = self.pxz_distribution.mean()

        self.pxz_distribution.param = self.decode(self.z2)
        rec_term_x2 = torch.mean(self.pxz_distribution.llkh(x2))
        rec_x2 = self.pxz_distribution.mean()
        
        return rec_x1, rec_x2, q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, variete_mu, variete_logvar, rec_term_x1, rec_term_x2
    
    def beta_anneal(self, epoch, beta_min=0.0, beta_max=1.0,  n_cycle=4, ratio=1, type = 'cycle_linear'):
        beta_val = np.ones(epoch) * beta_max
        period = epoch//n_cycle
        step = (beta_max-beta_min)/(period*ratio)
        if type=='cycle_linear':            
            for c in range(n_cycle):
                b, i = beta_min, 0
                while b <= beta_max and (int(i+c*period) < epoch):
                    beta_val[int(i+c*period)] = b
                    b += step
                    i += 1
        elif type=='cycle_sigmoid':
            for c in range(n_cycle):
                b, i = beta_min, 0
                while b <= beta_max:
                    beta_val[int(i+c*period)] = 1.0/(1.0+np.exp(-b))
                    b += step
                    i += 1
        elif type=='cycle_cosine':
            for c in range(n_cycle):
                b, i = beta_min, 0
                while b <= beta_max:
                    beta_val[int(i+c*period)] = 0.5-0.5*math.cos(math.pi*b)
                    b += step
                    i += 1
    
        return beta_val
    
    def loss_function(self, nb_epoch, epoch, 
                      q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, rec_term_x1, rec_term_x2):
        """
        Define loss function (elbo) = Reconstruction term + Kullback-Leiber Divergence term 
        """
        with torch.autograd.set_detect_anomaly(True):
            
            z1_kld = torch.mean((-0.5)*torch.mean(1 + q_z1_logvar - q_z1_mu**2 - q_z1_logvar.exp(), dim = 1))
            z2_kld = torch.mean((-0.5)*torch.mean(1 + q_z2_logvar - q_z2_mu**2 - q_z2_logvar.exp(), dim = 1))

            if self.beta_annealing:
                self.beta_val = self.beta_anneal(nb_epoch)
                loss = rec_term_x1 + rec_term_x2 - self.beta_val[epoch]*(z1_kld + z2_kld)
            else:
                loss = rec_term_x1 + rec_term_x2 - self.beta*(z1_kld + z2_kld)
            loss_dict = {"Loss": loss, "rec_term_1": rec_term_x1, "rec_term_2": rec_term_x2, 
                         "kld_1": z1_kld, "kld_2": z2_kld}
        
        return loss, loss_dict
    
    def step(self, x1, x2, nb_epochs, epoch, optimizer = None, train=False):
        if train:
            optimizer.zero_grad()

        rec_x1, rec_x2, q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, \
            variete_mu, variete_logvar, rec_term_x1, rec_term_x2 = self.forward(x1, x2, train=train)

        loss, loss_dict = self.loss_function(nb_epochs, epoch, q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar,
                                             rec_term_x1, rec_term_x2)
        if train:
            (-loss).backward()
            optimizer.step()
        return loss, loss_dict
    
    # Training one epoch
    def train_one_epoch(self, train_loader, nb_epochs, epoch, optimizer, device, shuffle_pairs = True):
        self.train()
        train_loss = 0.0
        total_rec_term1, total_rec_term2, total_z1_kld, total_z2_kld = 0, 0, 0, 0
        
        for i, (x1, x2, geno_ids, env1, env2) in enumerate(train_loader):
            
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()


            loss, loss_dict = self.step(x1, x2, nb_epochs, epoch, optimizer,train=True)
            train_loss += loss.item()
            total_rec_term1 += loss_dict["rec_term_1"]
            total_rec_term2 += loss_dict["rec_term_2"]
            total_z1_kld += loss_dict["kld_1"]
            total_z2_kld += loss_dict["kld_2"]

        if shuffle_pairs:
            train_loader.dataset.shuffle_data()

        
        return train_loss, loss_dict, total_rec_term1, total_rec_term2, total_z1_kld, total_z2_kld
    
    # Train function
    def train_(self, train_loader, train_data, train_gt, 
               optimizer, nb_epochs, device, print_rate, shuffle_pairs = True, lr_decay = False, step_size = 10,
               factor = 0.1, model_save_path='saved_models/best_model.pth'):
        """
        Train the Disentangled-VAE with the train data, then apply the model to the valid data to predict the missing
        NIRS of the validation data.
        train_loader: train dataloader
        valid_loader: valid dataloader
        train_data: train dataset for missing NIRS prediction
        valid_data: validation dataset for missing NIRS prediction
        train_gt: train groundtruth (the missing NIRS of the train data)
        valid_gt: validation groundtruth (the missing NIRS of the validation data)
        optimizer: optimization algorithm
        nb_epochs: number of epochs
        device: device used to train
        print_rate: number of epochs for printing the results
        shuffle_pairs: shuffle nirs pairs after each training epoch
        lr_decay: learning rate decay
        early_stopping: early stopping in the training or not
        patience: number of epochs to wait for improvement before stopping
        """
        loss_train, mse_train = [], []
        train_loss_dict = {}

        best_train_mse = float('inf')
        best_epoch = 0

        if lr_decay:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)

        for epoch in range(nb_epochs):
            ## Reconstruction
            # Training
            train_loss, loss_dict_new, rec_term1_train, rec_term2_train, kld1_train, kld2_train = \
                self.train_one_epoch(train_loader, nb_epochs, epoch, optimizer, device, shuffle_pairs)
            loss_train.append(train_loss/len(train_loader))
            train_loss_dict = append_elbo_dict(train_loss_dict, loss_dict_new)

            
            if (epoch + 1) % print_rate == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{nb_epochs}]:\n"+ "Reconstruction:\n" + "Train:"+
                    f" ELBO: {train_loss/len(train_loader):.4f},"+
                    f" Reconstruction term 1: {rec_term1_train/len(train_loader):.4f}," +
                    f" Reconstruction term 2: {rec_term2_train/len(train_loader):.4f}," +
                    f" KLD term 1: {kld1_train/len(train_loader):.4f}," +
                    f" KLD term 2: {kld2_train/len(train_loader):.4f}.",
                    flush=True
                )

            # Prediction
            # Train set
            train_mse = 0
            num_recon = 0
            self.eval()
            variete_list = train_data.GenoID.unique()  # list of variete
            env_list = train_data.Environnement.unique()  # list of Environment
            ### rewrite to be optimized
            for tar_variete in variete_list:
                # list of environment that having nirs for the target variete
                env_have = train_data[train_data.GenoID == tar_variete].Environnement.unique() 
                if len(env_have) == 0:
                    print(f'No NIRS available for the {tar_variete}')

                # list of environments that missing nirs for the target variete
                env_missing = [env for env in env_list if env not in env_have]

                for tar_env in env_missing:
                    # Reconstruction nirs of (tar_variete, env) from (src_variete, env)
                    rec_tar_nirs = reconstruct_one(self, train_data, tar_variete, tar_env, device)
                    #nirs_gt = np.array(train_gt[(train_gt.GenoID==tar_variete)&
                    #                             (train_gt.Environnement==tar_env)].iloc[:,2:])
                    nirs_gt = torch.tensor(train_gt[(train_gt.GenoID==tar_variete)&
                                                 (train_gt.Environnement==tar_env)].iloc[:,2:].values,
                                                 device=device)
                    if len(nirs_gt) == 0:
                        #print(f'No NIRS groundtruth available for the {tar_variete} in the environment {tar_env}')
                        continue
                    else:
                        nirs_gt = nirs_gt[0]

                    mse_loss = torch.mean((nirs_gt - rec_tar_nirs) ** 2).detach().cpu().numpy()
                    #mse_loss = mean_squared_error(nirs_gt, rec_tar_nirs)
                    train_mse += mse_loss
                    num_recon += 1
            
            mse_train.append((train_mse/num_recon))

            if (epoch + 1) % print_rate == 0 or epoch == 0:
                print("Predict missing NIRS:\n" + "Train set:"+
                    f" MSE: {(mse_train[-1]):.7f}\n" +
                    f"min MSE: {min(mse_train)}.")
                
            # Early stopping
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_epoch = epoch
                # Save the current best model state
                torch.save(self.state_dict(), model_save_path)
            
            if lr_decay:
                scheduler.step()
                
        return loss_train, train_loss_dict, mse_train, best_epoch
    
    def test_(self, test_miss, test_gt, device):
        recon = reconstruct_all_dvae(self, test_miss, device)
        mse_test, mse_df = mse_miss_recon(recon, test_gt)
        return mse_test, mse_df
    

