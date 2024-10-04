import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
#from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
import torch.optim as optim

from utils.append_dict import append_elbo_dict
from utils.reconstruction_mlvae import reconstruct_one, reconstruct_all_mlvae, mse_miss_recon

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim)

        # initialize weights
        #torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
 
    def forward(self, x):
        return self.linear_layer(x)
 
class MLVAE(nn.Module):
    """
    Multi-Level Variational AutoEncoder:
    pxz_distribution:
    input_dim: input size
    hidden_dims: list of hidden layer size
    latent_dim: latent size = size of latent vector for variete + size of latent vector for environment
    vari_latent_size: size of latent vector for variete
    """

    def __init__(self, pxz_distribution, input_dim, hidden_dims,
                 latent_dim, vari_latent_size, beta_v = 1, beta_e = 1,
                 beta_annealing = False, anneal_type = 'cycle_linear'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vari_latent_size = vari_latent_size
        self.pxz_distribution = pxz_distribution(input_dim)
        self.beta_v = beta_v
        self.beta_e = beta_e
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
                    LinearNorm(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
            )

        self.enc_mlp = nn.Sequential(*modules)
        self.enc_linear = LinearNorm(hidden_dims[-1], 2*latent_dim)
        self.variete = LinearNorm(2*latent_dim, self.vari_latent_size*2)
        self.env = LinearNorm(2*latent_dim, (latent_dim - self.vari_latent_size)*2)

        # Build decoder
        modules = []

        for in_dim, out_dim in zip([latent_dim] + hidden_dims[::-1], hidden_dims[::-1]):
            modules.append(
                nn.Sequential(
                    LinearNorm(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU())
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
        
    def accumulate_group_evidence(self, class_mu, class_logvar, labels_batch, device):
        """
        class_mu: mu values for class latent of each sample in the mini-batch
        class_logvar: logvar values for class latent for each sample in the mini-batch
        labels_batch: class labels of each sample
        """
        var_dict = {}
        mu_dict = {}

        # convert logvar to variance for calculations
        class_var = class_logvar.exp()

        # calculate var inverse for each group using group vars
        for i in range(len(labels_batch)):
            group_label = labels_batch[i]

            # remove 0 values from variances
            class_var = torch.where(class_var == 0, torch.full_like(class_var, 1e-6), class_var)

            if group_label in var_dict.keys():
                var_dict[group_label] += (1 / class_var[i])
            else:
                var_dict[group_label] = (1 / class_var[i])

        # invert var inverses to calculate mu and return value
        for group_label in var_dict.keys():
            var_dict[group_label] = 1 / var_dict[group_label]

        # calculate mu for each group
        for i in range(len(labels_batch)):
            group_label = labels_batch[i]

            if group_label in mu_dict.keys():
                mu_dict[group_label] += (class_mu[i] * (1 / class_var[i]))
            else:
                mu_dict[group_label] = (class_mu[i] * (1 / class_var[i]))

        # multiply group var with sums calculated above to get mu for the group
        for group_label in mu_dict.keys():
            mu_dict[group_label] *= var_dict[group_label]

        # replace individual mu and logvar values for each sample with group mu and logvar
        group_mu = torch.Tensor(class_mu.shape[0], class_mu.shape[1])
        group_var = torch.Tensor(class_var.shape[0], class_var.shape[1])

        group_mu.to(device)
        group_var.to(device)

        for i in range(len(labels_batch)):
            group_label = labels_batch[i]

            group_mu[i] = mu_dict[group_label]
            group_var[i] = var_dict[group_label]

            # remove 0 from var before taking log
            group_var[i] = torch.where(group_var[i] == 0, torch.full_like(group_var[i], 1e-6), group_var[i])

        # convert group vars into logvars before returning
        return group_mu, torch.log(group_var)
    
    def group_wise_reparameterize(self, mu, logvar, labels_batch, train=True):
        eps_dict = {}

        # generate only 1 eps value per group label
        for label in np.unique(labels_batch):
            eps_dict[label] = torch.randn_like(logvar[0])

        if train:
            std = torch.exp(torch.mul(logvar, 0.5))
            z = torch.zeros_like(std)

            # multiply std by correct eps and add mu
            for i in range(z.shape[0]):
                z[i] = eps_dict[labels_batch[i]] * std[i] + mu[i]
                
            return z
        else:
            return mu

    def forward(self, x, vars, device, train=True):
        variete_mu, variete_logvar, env_mu, env_logvar = self.encode(x)
        grouped_mu, grouped_logvar = self.accumulate_group_evidence(variete_mu, variete_logvar, vars, device)
        z_env = self.reparameterize(env_mu, env_logvar, train=train)
        z_variete = self.group_wise_reparameterize(grouped_mu, grouped_logvar, vars, train=train)
        z_variete = z_variete.to(device)
        z_env = z_env.to(device)
        z = torch.cat((z_variete, z_env), dim=-1)

        # Reconstruction
        self.pxz_distribution.param = self.decode(z)
        rec_term = torch.mean(self.pxz_distribution.llkh(x))
        rec_x = self.pxz_distribution.mean()

        return variete_mu, variete_logvar, env_mu, env_logvar, rec_term, rec_x
    
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
    
    def loss_function(self, nb_epoch, epoch, variete_mu, variete_logvar, env_mu, env_logvar, rec_term):
        """
        Define loss function (elbo) = Reconstruction term + Kullback-Leiber Divergence term 
        """
        with torch.autograd.set_detect_anomaly(True):
            
            kld_v = torch.mean((-0.5)*torch.mean(1 + variete_logvar - variete_mu.pow(2) - variete_logvar.exp(), dim = 1))
            kld_e = torch.mean((-0.5)*torch.mean(1 + env_logvar - env_mu.pow(2) - env_logvar.exp(), dim = 1))

            if self.beta_annealing:
                self.beta_val = self.beta_anneal(nb_epoch)
                loss = rec_term - self.beta_val[epoch]*(kld_v + kld_e)
            else:
                loss = rec_term - self.beta_v*kld_v - self.beta_e*kld_e
            loss_dict = {"Loss": loss, "rec_term": rec_term, "kld_v": kld_v, "kld_e": kld_e}

        return loss, loss_dict
    
    def step(self, x, nb_epochs, epoch, vars, device, optimizer = None, train=False):
        if train:
            optimizer.zero_grad()

        variete_mu, variete_logvar, env_mu, env_logvar, rec_term, rec_x = self.forward(x, vars, device, train=train)

        loss, loss_dict = self.loss_function(nb_epochs, epoch, variete_mu, variete_logvar, env_mu, env_logvar, rec_term)
        if train:
            (-loss).backward(retain_graph = True)
            optimizer.step()
        return loss, loss_dict
    
    # Training one epoch
    def train_one_epoch(self, train_loader, nb_epochs, epoch, optimizer, device):
        self.train()
        train_loss = 0.0
        total_rec_term, total_kld_v, total_kld_e = 0, 0, 0
        
        for i, (x, vars, envs) in enumerate(train_loader):
            
            x = x.to(device).float()
            
            loss, loss_dict = self.step(x, nb_epochs, epoch, vars, device, optimizer,train=True)
            train_loss += loss.item()
            total_rec_term += loss_dict["rec_term"]
            total_kld_v += loss_dict["kld_v"]
            total_kld_e += loss_dict["kld_e"]
        
        return train_loss, loss_dict, total_rec_term, total_kld_v, total_kld_e
    
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
            train_loss, loss_dict_new, rec_term_train, kld_v_train, kld_e_train= \
                self.train_one_epoch(train_loader, nb_epochs, epoch, optimizer, device)
            loss_train.append(train_loss/len(train_loader))
            train_loss_dict = append_elbo_dict(train_loss_dict, loss_dict_new)

            
            if (epoch + 1) % print_rate == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{nb_epochs}]:\n"+ "Reconstruction:\n" + "Train:"+
                    f" ELBO: {train_loss/len(train_loader):.4f},"+
                    f" Reconstruction term: {rec_term_train/len(train_loader):.4f}," +
                    f" KLD_v term: {kld_v_train/len(train_loader):.4f}," +
                    f" KLD_e term: {kld_e_train/len(train_loader):.4f}.",
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
        recon = reconstruct_all_mlvae(self, test_miss, device)
        mse_test, mse_df = mse_miss_recon(recon, test_gt)
        return mse_test, mse_df
