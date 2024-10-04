import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
#from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torch.optim as optim

from utils.append_dict import append_elbo_dict
from utils.reconstruction_cvae import reconstruct_one, reconstruct_all_cvae, mse_miss_recon

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
 
class CVAE(nn.Module):
    """
    Conditional Variational AutoEncoder:
    pxz_distribution:
    input_dim: input size (number of wavelengths)
    cond_dim: conditional dimension
    hidden_dims: list of hidden layer size
    latent_dim: latent size
    """

    def __init__(self, pxz_distribution, input_dim, cond_dim, hidden_dims, latent_dim, 
                 beta = 1, beta_annealing = False, anneal_type = 'cycle_linear'):
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.pxz_distribution = pxz_distribution(self.input_dim)
        self.beta = beta
        self.beta_annealing = beta_annealing
        if beta_annealing:
            self.anneal_type = anneal_type
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]

        # Build Encoder

        for in_dim, out_dim in zip([self.input_dim + self.cond_dim] + self.hidden_dims, self.hidden_dims):
            modules.append(
                nn.Sequential(
                    LinearInit(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
            )

        self.enc_mlp = nn.Sequential(*modules)
        self.enc_linear = LinearInit(self.hidden_dims[-1], 2*self.latent_dim)

        # Build decoder
        modules = []

        for in_dim, out_dim in zip([self.latent_dim + self.cond_dim] + self.hidden_dims[::-1], self.hidden_dims[::-1]):
            modules.append(
                nn.Sequential(
                    LinearInit(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
            )
        
        self.dec_mlp = nn.Sequential(*modules)
        # last linear layer to get decoder output
        self.dec_linear = LinearInit(self.hidden_dims[::-1][-1], self.pxz_distribution.decoder_output_dim) 

        self.mu = None
        self.logvar = None

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
        mu_logvar = self.enc_linear(output)
    
        mu = mu_logvar[:, : self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim : ]
        
        self.mu = mu
        self.logvar = logvar

        return mu, logvar
    
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
    
    def forward(self, x, cond, train=True):
        x_cond = torch.cat((x,cond), dim = 2)
        mu, logvar = self.encode(x_cond)
        z = self.reparameterize(mu, logvar, train = train)
        z_cond = torch.cat((z,cond.squeeze()), dim = 1)
        self.pxz_distribution.param = self.decode(z_cond)
        return self.pxz_distribution.param
    
    def beta_anneal(self, epoch, beta_min=0.0, beta_max=1.0,  n_cycle=4, ratio=1, type = 'cycle_linear'):
        '''
        Beta-anneling schedule
        '''
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
    
    def loss_function(self, nb_epoch, epoch, x):
        """
        Define loss function (elbo) = Reconstruction term + Kullback-Leiber Divergence term 
        """
        with torch.autograd.set_detect_anomaly(True):
            
            rec_term = self.pxz_distribution.llkh(x)
            rec_term = torch.mean(rec_term)
            kld = torch.mean((-0.5)*torch.mean(1 + self.logvar - self.mu**2 - self.logvar.exp(), dim = 1))

            if self.beta_annealing:
                self.beta_val = self.beta_anneal(nb_epoch)
                loss = rec_term - self.beta_val[epoch]*kld
            else:
                loss = rec_term - self.beta*kld
            loss_dict = {"Loss": loss, "rec_term": rec_term, "kld": kld}
        
        return loss, loss_dict
    
    def step(self, x, cond, nb_epochs, epoch, optimizer = None, train=False):
        if train:
            optimizer.zero_grad()

        self.forward(x, cond, train=train)
        loss, loss_dict = self.loss_function(nb_epochs, epoch, x)

        if train:
            (-loss).backward()
            optimizer.step()
        return loss, loss_dict
    
    # Training one epoch
    def train_one_epoch(self, train_loader, nb_epochs, epoch, optimizer, device):
        self.train()
        train_loss = 0.0
        total_rec_term, total_kld = 0, 0
        
        for i, (x, geno,env,_,_) in enumerate(train_loader):
            
            x = x.to(device).float()
            geno = geno.to(device).float()
            env = env.to(device).float()
            
            geno = geno.view(geno.shape[0],1, -1)
            env = env.view(env.shape[0],1, -1)
            cond = torch.cat((geno,env), dim = 2)

            loss, loss_dict = self.step(x, cond, nb_epochs, epoch, optimizer,train=True)
            train_loss += loss.item()
            total_rec_term += loss_dict["rec_term"]
            total_kld += loss_dict["kld"]

        return train_loss, loss_dict, total_rec_term, total_kld
    
    # Train function
    def train_(self, train_loader, train_data, train_gt, 
               optimizer, nb_epochs, device, print_rate, lr_decay = False, step_size = 10,
               factor = 0.1, model_save_path='saved_models/best_model.pth'):
        """
        Train the Conditional VAE with the train data, then apply the model predict the missing
        NIRS of the validation data.
        train_loader: train dataloader
        train_data: train dataset for missing NIRS prediction
        train_gt: train groundtruth (the missing NIRS of the train data)
        optimizer: optimization algorithm
        nb_epochs: number of epochs
        device: device used to train
        print_rate: number of epochs for printing the results
        lr_scheduler: learning rate scheduler
        """
        loss_train, mse_train = [], []
        train_loss_dict = {}
        #loss_valid, mse_valid = [], []
        #valid_loss_dict = {}

        best_train_mse = float('inf')
        best_epoch = 0

        if lr_decay:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)

        for epoch in range(nb_epochs):
            ## Reconstruction
            # Training
            train_loss, loss_dict_new, rec_term_train, kld_train = \
                self.train_one_epoch(train_loader, nb_epochs, epoch, optimizer, device)
            loss_train.append(train_loss/len(train_loader))
            train_loss_dict = append_elbo_dict(train_loss_dict, loss_dict_new)

            
            if (epoch + 1) % print_rate == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{nb_epochs}]:\n"+ "Reconstruction:\n" + "Train:"+
                    f" ELBO: {train_loss/len(train_loader):.4f},"+
                    f" Reconstruction term 1: {rec_term_train/len(train_loader):.4f}," +
                    f" KLD term: {kld_train/len(train_loader):.4f}.",
                    flush=True
                )

            # Prediction
            # Train set
            train_mse = 0
            num_recon = 0
            self.eval()
            variete_list = train_data.GenoID.unique()  # list of variete
            env_list = train_data.Environnement.unique()  # list of Environment

            for tar_variete in variete_list:
                # list of environment that having nirs for the target variete
                env_have = train_data[train_data.GenoID == tar_variete].Environnement.unique()
                if len(env_have) == 0:
                    print(f'No NIRS available for the {tar_variete}')

                # list of environments that missing nirs for the target variete
                env_missing = [env for env in env_list if env not in env_have]

                for tar_env in env_missing:
                    # Reconstruction nirs of (tar_variete, tar_env)
                    tar_var_code, tar_env_code = train_loader.dataset.get_code(tar_variete, tar_env)
                    rec_tar_nirs = reconstruct_one(self, tar_var_code, tar_env_code, device)
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
            
            # Learning rate scheduling
            if lr_decay:
                scheduler.step()

            # Save the state of the model with the lowest MSE 
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_epoch = epoch
                # Save the current best model state
                torch.save(self.state_dict(), model_save_path)

            """ if (early_stopping == True) & (epochs_without_improvement >= patience):
                print(f"Early stopping at epoch {epoch + 1}")
                break """

        return loss_train, train_loss_dict, mse_train, best_epoch
    
    def test_(self, test_miss, test_gt, device):
        recon = reconstruct_all_cvae(self, test_miss, device)
        mse_test, mse_df = mse_miss_recon(recon, test_gt)
        return mse_test, mse_df
    
    


