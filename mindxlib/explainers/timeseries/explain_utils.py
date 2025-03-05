import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import copy
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pickle
import torch.utils.data as Data
import time


class Dataset_Explain(Dataset):
    def __init__(self, data) -> None: # data: numpy array, #sample X #feature X seq_len
        super(Dataset_Explain,self).__init__()
        self.num_samples, self.num_features, self.seq_len = data.shape
        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        seq_x = self.data_x[index,:,:]
        seq_y = self.data_y[index,:,:]
        mask = torch.zeros((self.num_features,self.seq_len)).to(seq_y.device)
        pos = torch.sort(torch.randperm(self.seq_len+1)[:2])[0]
        mask[torch.randperm(self.num_features)[:self.num_features//2+1],pos[0]:pos[1]] = 1

        return seq_x, mask, seq_y

    def __len__(self):
        return self.num_samples


class res_block(nn.Module):
    def __init__(self, input_size, output_size, BN_enable=False):
        super(res_block, self).__init__()
        self.linear_layer = nn.Linear(input_size, output_size)
        self.res_layer = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.BN_enable = BN_enable
        
    def forward(self, x):

        x = x.to(torch.float32)
        if self.BN_enable:
            return F.relu(self.bn(self.linear_layer(x))) + self.res_layer(x)
        else:
            return F.relu(self.linear_layer(x)) + self.res_layer(x)

    
class ImpVAE(nn.Module):
    def __init__(self,num_features, seq_len,layer_dim, device, BN_enable=False):
        super(ImpVAE, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.input_dims = num_features*seq_len
        self.enc_latent_dims = layer_dim[-1]
        self.device = device 
        layers = []
        
        layers.append(res_block(self.input_dims*2,layer_dim[0],BN_enable=False))
        for ii in range(len(layer_dim)-2):
            layers.append(res_block(layer_dim[ii],layer_dim[ii+1],BN_enable))
        layers.append(res_block(layer_dim[-2],2*layer_dim[-1],BN_enable=False))
        self.encoder = nn.Sequential(*layers)
        layers = []
        layers.append(res_block(layer_dim[-1],layer_dim[-2],BN_enable=False))
        for ii in range(len(layer_dim)-2,0,-1):
            layers.append(res_block(layer_dim[ii],layer_dim[ii-1],BN_enable))
        layers.append(res_block(layer_dim[0],self.input_dims,BN_enable=False))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, mask): # x: batch_size X num_features X seq_len
        x_flatten = torch.flatten(x, start_dim=1) # x_hat: batch_size X (num_features * seq_len)
        mask_flatten = torch.flatten(mask, start_dim=1)
       
        z = self.encoder(torch.cat((x_flatten,mask_flatten),dim=-1))
        mu = z[:,:self.enc_latent_dims]
        sigma = z[:,self.enc_latent_dims:]
        z_sample = mu + sigma*torch.randn(sigma.shape).to(self.device)
        x_hat = self.decoder(z_sample).reshape(-1,self.num_features,self.seq_len)
        return x_hat, mu, sigma
    
    def generate(self, x, mask, num_sample):
        self.eval()
        batch_size = x.shape[0]
        x_flatten = torch.flatten(x, start_dim=1) # x_hat: batch_size X (num_features * seq_len)
        mask_flatten = torch.flatten(mask, start_dim=1)

        z = self.encoder(torch.cat((x_flatten,mask_flatten),dim=-1))
        mu = z[:,:self.enc_latent_dims]
        sigma = z[:,self.enc_latent_dims:]
        z_sample = mu + sigma.repeat_interleave(num_sample,dim=0)*torch.randn(batch_size*num_sample,self.enc_latent_dims).to(self.device)
        x_hat = self.decoder(z_sample).reshape(-1,self.num_features,self.seq_len)
        return x_hat

    def train_model(self,train_dataset,lr=0.01,batch_size=320,n_epochs=500,vae_train_print=False):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.7)
        for epoch in range(n_epochs):
            train_loss = 0.0
            self.train()
            for step,(X, mask, Y) in enumerate(train_loader):
                X_hat, mu, sigma = self.forward(X*mask, mask)
                recons_loss = torch.mean(torch.mean(torch.sum((Y - X_hat) ** 2, dim=-1), dim=-1),dim=0)
                reg_loss = torch.mean(torch.mean(torch.sum(((Y - X_hat)*mask) ** 2, dim=-1), dim=-1),dim=0)
                reg = torch.mean(torch.mean(mu ** 2 , dim=0),dim=0) + torch.mean(torch.mean(torch.abs(sigma)** 2, dim=0),dim=0) - torch.mean(torch.mean(torch.log(sigma**2+1e-10), dim=0),dim=0)
                loss = recons_loss + 1*reg + 0.5*reg_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                optimizer.step()
                train_loss += recons_loss.item()*X.shape[0]
            scheduler.step()
            train_loss = train_loss
            if  vae_train_print:
                print(f'Epoch: {epoch+1} Train mse = {np.round(train_loss,4)}')
        

class PatchAttributionTorch():
    """
    This class provides an explanation method for time series.
    The input of model is defined in function 'model_predict', the input 
    """
    def __init__(self, func, patch_size, x_size, is_numpy_model, generator, device, only_last, 
                 sample_num=10, lambda_1=0, kk=1e+15):
        self.func = func  # Function to be explained, expected input format: batch_size X num_features X seq_len
        self.output_dim = None
        self.patch_size = patch_size
        self.device = device
        self.x_size = x_size
        self.patch_num = np.ceil(self.x_size / self.patch_size)
        self.generator = generator.to(self.device)
        self.is_numpy_model = is_numpy_model  # Whether the function to be explained is a numpy model
        self.sample_num = sample_num  # Number of samples used when calculating expectation
        self.lambda_1 = lambda_1
        self.kk = kk
        self.only_last = only_last
        
    def generate_samples(self, x, cared_fid, ii, jj):
        self.generator.eval()

        x = x.unsqueeze(0)

        mask = torch.zeros_like(x).to(self.device)
        mask[:, cared_fid, ii * self.patch_size:(jj + 1) * self.patch_size] = 1

        X_hat = self.generator.generate(x * mask, mask, self.sample_num)

        X_hat = X_hat.detach()
        return X_hat

    def compute_weight(self, ii):
        weight = torch.exp(-self.lambda_1 * torch.arange(ii) * (ii - 1 - torch.arange(ii)) / ii)
        return weight / torch.sum(weight)

    def model_predict(self, x):
        """
        input: #samples X #feature X #seq_len
        output: #sample X #output
        """
        
        return self.func(x)

    def model_predict_w_replace(self, x, cared_fid, replace_x):
        """
        x: #feature X #seq_len
        replace_x: #sample X #num_cared_fid X #seq_len
        result: #sample X #output X #seq_len
        """
        num_feature, seq_len = x.shape
        num_sample = replace_x.shape[0]
        x_hat = torch.repeat_interleave(x.reshape(1, num_feature, seq_len), num_sample, axis=0)
        x_hat[:, cared_fid, :] = replace_x
        result = self.model_predict(x_hat)

        if len(result.shape) == 2:
           result = result.unsqueeze(-1)

        return result

    def compute_interactions(self, x, cared_fid, time_step):
        interaction_matrix = None
        x_cared = x[cared_fid, :]

        generated_samples = {}
        KK = self.kk
        for ii in range(int(self.patch_num)):
            for jj in range(ii, min(ii + KK + 1, int(self.patch_num))):
                # generated_samples: #num_samples X #num_cared_fid X #seq_len
                generated_samples[(ii, jj)] = self.generate_samples(x, cared_fid, ii, jj)[:, cared_fid, :]

        for ii in range(int(self.patch_num)):
            conter_samples_0 = generated_samples[(ii, ii)]
            mixed_ii = copy.deepcopy(conter_samples_0)
            mixed_ii[:, :, ii * self.patch_size:(ii + 1) * self.patch_size] = x_cared[:, ii * self.patch_size:(ii + 1) * self.patch_size]
            margs_ii = self.model_predict_w_replace(x, cared_fid, mixed_ii)[:,:,time_step].mean(dim=0)

            if interaction_matrix is None:
                interaction_matrix = torch.zeros((int(self.patch_num), int(self.patch_num), margs_ii.shape[-1])).to(self.device)
            interaction_matrix[ii, ii, :] = margs_ii
            if ii < int(self.patch_num) - 1:
                conter_samples_1 = generated_samples[(ii + 1, ii + 1)]
                mixed_ii = copy.deepcopy(conter_samples_1)
                mixed_ii[:, :, ii * self.patch_size:(ii + 2) * self.patch_size] = x_cared[:, ii * self.patch_size:(ii + 2) * self.patch_size]
                mixed_ii[:, :, :ii * self.patch_size] = conter_samples_0[:, :, :ii * self.patch_size]
                margs_ii = self.model_predict_w_replace(x, cared_fid, mixed_ii)[:,:,time_step].mean(dim=0)
                interaction_matrix[ii, ii + 1, :] = margs_ii
            for jj in range(ii + 2, min(ii + KK, int(self.patch_num))):
                conter_samples_0 = generated_samples[(ii + 1, jj - 1)]
                conter_samples_1 = generated_samples[(ii, jj - 1)]
                conter_samples_2 = generated_samples[(ii + 1, jj)]
                mixed_ij_0 = copy.deepcopy(conter_samples_0)
                mixed_ij_0[:, :, (ii + 1) * self.patch_size:jj * self.patch_size] = x_cared[:, (ii + 1) * self.patch_size:jj * self.patch_size]

                mixed_ij_3 = copy.deepcopy(conter_samples_1)
                mixed_ij_3[:, :, ii * self.patch_size:(jj + 1) * self.patch_size] = x_cared[:, ii * self.patch_size:(jj + 1) * self.patch_size]
                mixed_ij_3[:, :, (jj + 1) * self.patch_size:] = conter_samples_2[:, :, (jj + 1) * self.patch_size:]

                mixed_ij_03 = torch.cat((mixed_ij_0, mixed_ij_3), dim=0)
                margs_03 = self.model_predict_w_replace(x, cared_fid, mixed_ij_03)[:,:,time_step].mean(dim=0)

                mixed_ij_1 = copy.deepcopy(conter_samples_0)
                mixed_ij_1[:, :, ii * self.patch_size:jj * self.patch_size] = x_cared[:, ii * self.patch_size:jj * self.patch_size]
                mixed_ij_1[:, :, :ii * self.patch_size] = conter_samples_1[:, :, :ii * self.patch_size]

                mixed_ij_2 = copy.deepcopy(conter_samples_0)
                mixed_ij_2[:, :, (ii + 1) * self.patch_size:(jj + 1) * self.patch_size] = x_cared[:, (ii + 1) * self.patch_size:(jj + 1) * self.patch_size]
                mixed_ij_2[:, :, (jj + 1) * self.patch_size:] = conter_samples_2[:, :, (jj + 1) * self.patch_size:]

                mixed_ij_12 = torch.cat((mixed_ij_1, mixed_ij_2), dim=0)
                margs_12 = self.model_predict_w_replace(x, cared_fid, mixed_ij_12)[:,:,time_step].mean(dim=0)

                if interaction_matrix is None:
                    interaction_matrix = torch.zeros((int(self.patch_num), int(self.patch_num), margs_03.shape[-1])).to(self.device)

                interaction_matrix[ii, jj, :] = 2 * (margs_03 - margs_12)

        baseline = torch.mean(torch.diagonal(interaction_matrix), axis=1)
        for ii in range(int(self.patch_num) - 1):
            interaction_matrix[ii, ii + 1, :] = interaction_matrix[ii, ii + 1, :] - interaction_matrix[ii, ii, :] - interaction_matrix[ii + 1, ii + 1, :] + baseline
            interaction_matrix[ii, ii, :] -= baseline
        interaction_matrix[-1, -1, :] -= baseline

        return interaction_matrix


    def compute_interactions_with_timesteps(self, x, cared_fid, time_steps):
        interaction_matrices_per_timestep = []
        
        if self.only_last:
            all_time_steps_interaction_matrix = self.compute_interactions(x, cared_fid, time_step=-1)
        else:

            for t in range(time_steps):

                interaction_matrix_t = self.compute_interactions(x, cared_fid, time_step=t)
                interaction_matrices_per_timestep.append(interaction_matrix_t)
            
            all_time_steps_interaction_matrix = torch.stack(interaction_matrices_per_timestep, dim=-1).to(self.device)
            
            
        self.interaction_matrix = all_time_steps_interaction_matrix
    def attribute(self, x, cared_fid, lambda_1=None):
        # x: #feature X #seq_len 
        # Model input is #batch_size X #feature X #seq_len
        # #num_patch X #output

        N = x.shape[1]
        self.patch_num = (N // self.patch_size)

        if self.patch_num * self.patch_size < N:
            self.patch_num += 1
        if lambda_1 is not None:
            self.lambda_1 = lambda_1
        if not isinstance(cared_fid, list):  # Enforce that cared_fid is a list
            cared_fid = [cared_fid]

        with torch.no_grad():
            self.compute_interactions_with_timesteps(x, cared_fid, N)
            # print("interaction_matrix shape",self.interaction_matrix.shape)
    

        if self.only_last:
            attributions = torch.zeros((int(self.patch_num), self.interaction_matrix.shape[-1]))
            for ii in range(int(self.patch_num)):
                for jj in range(ii, int(self.patch_num)):
                    if ii == jj:

                        attributions[ii:jj + 1, :] += self.compute_weight(jj - ii + 1).reshape(-1, 1).mm(self.interaction_matrix[ii, jj, :].reshape(1, -1).detach().cpu())
                    else:
                        attributions[ii:jj + 1, :] += self.compute_weight(jj - ii + 1).reshape(-1, 1).mm(self.interaction_matrix[ii, jj, :].reshape(1, -1).detach().cpu())
        else:
            attributions = torch.zeros((int(self.patch_num), self.interaction_matrix.shape[-2], N))
            for t in range(int(N)):
                for ii in range(int(self.patch_num)):
                    for jj in range(ii, int(self.patch_num)):
                        if ii == jj:

                            attributions[ii:jj + 1, :, t] += self.compute_weight(jj - ii + 1).reshape(-1, 1).mm(self.interaction_matrix[ii, jj, :, t].reshape(1, -1).detach().cpu())
                        else:
                            attributions[ii:jj + 1, :, t] += self.compute_weight(jj - ii + 1).reshape(-1, 1).mm(self.interaction_matrix[ii, jj, :, t].reshape(1, -1).detach().cpu())
        return attributions