import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
#from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from tqdm import tqdm

from utils.append_dict import append_elbo_dict
from utils.reconstruction_hvae import reconstruct_one, reconstruct_all, mse_miss_recon

class LinearNorm(nn.Module):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim)

        # He initialization
        nn.init.kaiming_normal_(self.linear_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.linear_layer.bias, 0.0)
 
    def forward(self, x):
        return self.linear_layer(x)
 
class HVAE(nn.Module):
    """
    Hierarchical Variational AutoEncoder:
    pxz_distribution:
    input_dim: input size
    hidden_dims: list of hidden layer size
    latent_dim: latent size = size of latent vector for variete + size of latent vector for environment
    vari_latent_size: size of latent vector for variete
    """

    def __init__(self, pxz_distribution, input_dim, hidden_dims,
                  latent_dim, vari_latent_size, 
                  var_v_init, var_e_init,
                  beta_v = 1, beta_e = 1, beta_ve = 1, beta_ev = 1,
                  beta_annealing = False, anneal_type = 'cycle_linear'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vari_latent_size = vari_latent_size
        self.var_v_init = var_v_init
        self.var_e_init = var_e_init
        self.pxz_distribution = pxz_distribution(input_dim)
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.beta_ve = beta_ve
        self.beta_ev = beta_ev
        self.beta_annealing = beta_annealing
        self.z_v_dict = {}
        self.z_e_dict = {}
        self.var_v = nn.Parameter(torch.tensor(self.var_v_init))
        self.var_e = nn.Parameter(torch.tensor(self.var_e_init))
        if beta_annealing:
            self.anneal_type = anneal_type
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]

        # Build Encoder

        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(
                nn.Sequential(
                    LinearNorm(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.SiLU())
            )

        self.enc_mlp = nn.Sequential(*modules)
        self.enc_linear = LinearNorm(hidden_dims[-1], 2*latent_dim)
        self.ve_mlp = LinearNorm(2*latent_dim, self.vari_latent_size*2)
        self.ev_mlp = LinearNorm(2*latent_dim, (latent_dim - self.vari_latent_size)*2)

        # Build decoder
        modules = []

        for in_dim, out_dim in zip([latent_dim] + hidden_dims[::-1], hidden_dims[::-1]):
            modules.append(
                nn.Sequential(
                    LinearNorm(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.SiLU())
            )
        
        self.dec_mlp = nn.Sequential(*modules)
        # last linear layer to get decoder output
        self.dec_linear = LinearNorm(hidden_dims[::-1][-1], self.pxz_distribution.decoder_output_dim) 

    def encode(self, x):
        """
        Encodes the input spectra x by passing through the encoder network
        and returns the mean and log variance of latent variables for variete and environment.
        x: input data
        ouput: mean (mu) and log variance (logvar) of latent variable z
        """

        shape = x.shape

        x = x.reshape(shape[0],-1)
        output = self.enc_mlp(x)
        output = self.enc_linear(output)
        ve = self.ve_mlp(output)
        ev = self.ev_mlp(output)

        ve_mu = ve[:,:self.vari_latent_size]
        ve_logvar = ve[:,self.vari_latent_size:]
        ev_mu = ev[:,:(self.latent_dim-self.vari_latent_size)]
        ev_logvar = ev[:,(self.latent_dim-self.vari_latent_size):]
        
        return ve_mu, ve_logvar, ev_mu, ev_logvar
    
    def decode(self, z):
        """
        Decodes the latent variable z by passing through the decoder network
        z: latent variable z
        """
        z = z.reshape(1,-1)
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
        
    def kld_normal(self, mu1, logvar1, mu2, logvar2):
        # Convert log-variances to variances
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
            
        # Calculate the KL divergence
        kld = torch.mean((-0.5)*torch.mean(1 - (var1/var2) - (logvar2-logvar1) - ((mu1 - mu2)**2) / var2, dim = 1))
            
        return kld
    
    def forward(self, x, vars, envs, dataloader, device, train=True):
        var_set = set(vars)
        env_set = set(envs)
        # Encode the spectrum to get z_ve, z_ev
        ve_mu, ve_logvar, ev_mu, ev_logvar = self.encode(x)
        z_ve = self.reparameterize(ve_mu, ve_logvar, train = train)
        z_ev = self.reparameterize(ev_mu, ev_logvar, train = train)

        rec_term = 0
        kld_ve, kld_v = 0,0
        kld_ev, kld_e = 0,0

        # Compute for varieties
        for v in var_set:
            indices = [idx for idx, val in enumerate(vars) if val == v]
            n_e = len(indices)

            v_mu = (1/(n_e+self.var_v))*torch.sum(z_ve[indices], dim = 0).to(device)
            v_logvar = torch.log((self.var_v/(n_e+self.var_v)).expand_as(v_mu)).to(device)

            z_v = self.reparameterize(v_mu, v_logvar, train = train)
            kld_ve += self.kld_normal(ve_mu[indices], ve_logvar[indices], 
                                     z_v.repeat(ve_mu[indices].shape[0],1), 
                                     torch.log(self.var_v).expand_as(z_v).repeat(ve_mu[indices].shape[0],1))
            kld_v += (-0.5)*torch.mean(1 + v_logvar - v_mu**2 - v_logvar.exp())
            self.z_v_dict[v] = z_v.clone().detach()
        
        # compute for environments
        for e in env_set:
            if e not in self.z_e_dict:
                z_e_old = torch.randn((z_ev.shape[1],), requires_grad = True).to(device)
            else:
                z_e_old = self.z_e_dict[e].clone().requires_grad_().to(device)

            indices = [idx for idx, val in enumerate(envs) if val == e]
            n_v = dataloader.dataset.get_num_var(e)

            ev_miss_logvar = torch.log((self.var_e/(n_v-len(indices))).expand_as(z_e_old)).to(device)
            z_ev_miss = self.reparameterize(z_e_old, ev_miss_logvar, train = train)

            e_mu = (1/(n_v + self.var_e))*torch.sum(z_ev[indices], dim = 0) + \
                ((n_v-len(indices))/(n_v + self.var_e))*z_ev_miss
            e_logvar = torch.log((self.var_e/(n_v+self.var_e)).expand_as(e_mu))
            e_mu = e_mu.to(device)
            e_logvar = e_logvar.to(device)
            z_e = self.reparameterize(e_mu, e_logvar, train = train)
            kld_ev += self.kld_normal(ev_mu[indices], ev_logvar[indices], 
                                     z_e.repeat(ev_mu[indices].shape[0],1), 
                                     torch.log(self.var_e).expand_as(z_e).repeat(ev_mu[indices].shape[0],1))
            kld_e += (-0.5)*torch.mean(1 + e_logvar - e_mu**2 - e_logvar.exp())
            self.z_e_dict[e] = z_e.clone().detach()

        # Compute reconstruction term
        z = torch.cat((z_ve, z_ev), dim = -1)
        self.pxz_distribution.param = self.decode(z)
        rec_term += torch.mean(self.pxz_distribution.llkh(x))
        
        rec_term /= x.shape[0]        
        kld_v /= len(var_set)
        kld_e /=  len(env_set)
        kld_ve /= len(var_set)
        kld_ev /= len(env_set)
        
        return rec_term, kld_v, kld_e, kld_ve, kld_ev
    
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
    
    def loss_function(self, nb_epoch, epoch, rec_term, kld_v, kld_e, kld_ve, kld_ev, train=False):
        """
        Define loss function (elbo) = Reconstruction term + Kullback-Leiber Divergence term 
        """
        with torch.autograd.set_detect_anomaly(True):

            if self.beta_annealing:
                self.beta_val = self.beta_anneal(nb_epoch)
                loss = rec_term - self.beta_val[epoch]*(kld_v + kld_e + kld_ev + kld_ve)
            else:
                loss = rec_term - self.beta_v*kld_v - self.beta_e*kld_e - self.beta_ve*kld_ve - self.beta_ev*kld_ev
            loss_dict = {"Loss": loss, "rec_term": rec_term, "kld_v": kld_v, "kld_e": kld_e, 
                         "kld_ev": kld_ev, "kld_ve": kld_ve}
        
        return loss, loss_dict
    
    def step(self, x, nb_epochs, epoch, vars, envs, dataloader, device, optimizer = None, train=False):
        if train:
            optimizer.zero_grad()

        rec_term, kld_v, kld_e, kld_ve, kld_ev = self.forward(x, vars, envs, dataloader, device, train=train)

        loss, loss_dict = self.loss_function(nb_epochs, epoch, rec_term, kld_v, kld_e, kld_ve, kld_ev, train = train)
        if train:
            (-loss).backward()
            optimizer.step()
        return loss, loss_dict
    
    # Training one epoch
    def train_one_epoch(self, train_loader, nb_epochs, epoch, optimizer, device):
        self.train()
        train_loss = 0
        total_rec_term, total_kld_v, total_kld_e, total_kld_ve, total_kld_ev = 0, 0, 0, 0, 0
        
        for x, vars, envs in train_loader: 
            x = x.to(device).float()
            vars = vars
            envs = envs
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            loss, loss_dict = self.step(x, nb_epochs, epoch, vars, envs, 
                                        train_loader, device, optimizer,train=True)
            train_loss += loss.item()
            total_rec_term += loss_dict["rec_term"]
            total_kld_v += loss_dict["kld_v"]
            total_kld_e += loss_dict["kld_e"]
            total_kld_ve += loss_dict["kld_ve"]
            total_kld_ev += loss_dict["kld_ev"]
        
        return train_loss, loss_dict, total_rec_term, total_kld_v, total_kld_e, total_kld_ve, total_kld_ev
    
    # Train function
    def train_(self, train_loader, train_data, train_gt, 
               optimizer, nb_epochs, device, print_rate, lr_decay = False, step_size = 10,
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

        for epoch in tqdm(range(nb_epochs)):
            ## Reconstruction
            # Training
            train_loss, loss_dict_new, rec_term_train, kld_v_train, kld_e_train, kld_ve_train, kld_ev_train = \
                self.train_one_epoch(train_loader, nb_epochs, epoch, optimizer, device)
            loss_train.append(train_loss/len(train_loader))
            train_loss_dict = append_elbo_dict(train_loss_dict, loss_dict_new)

            
            if (epoch + 1) % print_rate == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{nb_epochs}]:\n"+ "Reconstruction:\n" + "Train:"+
                    f" ELBO: {train_loss/len(train_loader):.4f},"+
                    f" Reconstruction term: {rec_term_train/len(train_loader):.4f}," +
                    f" KLD_v: {kld_v_train/len(train_loader):.4f}," +
                    f" KLD_e : {kld_e_train/len(train_loader):.4f}." +
                    f" KLD_ve : {kld_ve_train/len(train_loader):.4f}." +
                    f" KLD_ev : {kld_ev_train/len(train_loader):.4f}.",
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
                    rec_tar_nirs = reconstruct_one(self, tar_variete, tar_env, device)
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
        recon = reconstruct_all(self, test_miss, device)
        mse_test, mse_df = mse_miss_recon(recon, test_gt)
        return mse_test, mse_df
