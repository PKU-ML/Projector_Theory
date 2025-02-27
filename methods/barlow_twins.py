# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F

def mat_pow(matrix, alpha):
    n = matrix.shape[0]
    res = torch.eye(n).cuda()
    for i in range(alpha):
        res = res @ matrix
    return res

def renyi_entropy(matrix, alpha):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    
    # Calculate the Renyi entropy
    if alpha == 1:
       entropy = -torch.trace(matrix/n @ torch.log(matrix/n))
    else:
       entropy = 1 / (1-alpha) * torch.log(torch.trace(mat_pow(matrix/n, alpha)))
    return entropy

def matrix_mutual_information(Z1, Z2, alpha):
    matrix1 = Z1 @ Z1.T
    matrix2 = Z2 @ Z2.T

    # Calculate individual entropies
    h1 = renyi_entropy(matrix1, alpha)
    h2 = renyi_entropy(matrix2, alpha)
    
    # Calculate the Hadamard product of the two matrices
    hadamard_product = torch.mul(matrix1, matrix2)
    
    # Calculate the joint Renyi entropy
    joint_entropy = renyi_entropy(hadamard_product, alpha)

    # Calculate mutual information
    mutual_info = h1 + h2 - joint_entropy
    return mutual_info


class BarlowTwins(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
                scale_loss (float): scaling factor of the loss.
        """

        super().__init__(cfg)

        self.lamb: float = cfg.method_kwargs.lamb
        self.scale_loss: float = cfg.method_kwargs.scale_loss
        self.final_dim: int = cfg.final_dim
        self.point_num: int = cfg.point_num

        # regularization parameter
        self.lmbd = cfg.lmbd
        # renyi-entropy parameter
        self.alpha = cfg.alpha
        # sparse autoencoder parameter
        self.if_sparse = cfg.sparse_autoencoder
        self.topk = cfg.topk
        self.latents_dim = cfg.latents_dim

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.final_dim

        # projector
        # if True then employ the sparse autoencoder (can be emplemented with other two methods)
        if self.if_sparse :
            # preserve half of the original projector 
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
            )
            self.auto_encoder: nn.Module = nn.Linear(proj_hidden_dim, self.latents_dim)
            self.auto_decoder: nn.Module = nn.Linear(self.latents_dim, proj_hidden_dim)
            self.pre_bias =nn.Parameter(torch.zeros(proj_hidden_dim))
        # if False then just employ the original projector
        else :
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, self.final_dim),
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(BarlowTwins, BarlowTwins).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.0051)
        cfg.method_kwargs.scale_loss = omegaconf_select(cfg, "method_kwargs.scale_loss", 0.024)

        return cfg
    

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        if self.if_sparse :
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()},
                                      # add learnable params of sparse autoencoder
                                      {"name": "bias", "params": self.pre_bias},                      
                                      {"name": "autoencoder", "params": self.auto_encoder.parameters()},
                                      {"name": "autodecoder", "params": self.auto_decoder.parameters()},
                                     ]
        else :
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def round_ste(self, z):
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()
    def bound(self, z, discretization, eps: float = 1e-3):
        """Bound z, an array of shape ( ..., d)."""
        half_l = (discretization - 1) * (1 - eps) / 2
        shift1 = torch.tensor(0.5, dtype=torch .float32)
        shift2 = torch.tensor(0.0, dtype=torch .float32)
        if discretization % 2 == 0 :
            offset = shift1
        else:
            offset = shift2
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset
    
    def quantize(self, z, discretization):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z, discretization))
        half_width = discretization // 2 # Renormalize to [-1， 1]
        return quantized / half_width

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z2 = self.projector(out["z1"])
        if self.if_sparse :
            # employ the sparse autoencoder as projector
            z2 = self.sparse_autoencoder(z2 - self.pre_bias)
            z2 = z2 + self.pre_bias
        if self.point_num != 0:
            z2 = self.quantize(z2, self.point_num)
        out.update({"z2": z2})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        self.class_loss = out["loss"]
        self.acc = out["acc"]
        proj1, proj2 = out["z2"]
        enc1, enc2 = out["z1"]

        # ------- barlow twins loss -------
        self.barlow_loss = barlow_loss_func(proj1, proj2, lamb=self.lamb, scale_loss=self.scale_loss)
        self.encoder_barlow_loss = barlow_loss_func(enc1, enc2, lamb=self.lamb, scale_loss=self.scale_loss)

        # calculate upper and lower bound using encoder and decoder features
        Z1 = F.normalize(torch.cat(out["z1"]))
        Z2 = F.normalize(torch.cat(out["z2"]))
        self.reg_loss = matrix_mutual_information(Z1, Z2, self.alpha)
        self.encoder_mat_info = renyi_entropy(Z1 @ Z1.T, self.alpha)
        self.projector_mat_info = renyi_entropy(Z2 @ Z2.T, self.alpha)
        self.upper_bound = self.encoder_mat_info - self.reg_loss
        self.lower_bound = -self.encoder_barlow_loss - self.reg_loss


        # recorde the vital params during training
        self.log("train_barlow_loss", self.barlow_loss, on_epoch=True, sync_dist=True)
        self.log("train_class_loss", self.class_loss, on_epoch=True, sync_dist=True)
        self.log("train_reg_loss", self.reg_loss, on_epoch=True, sync_dist=True)
        self.log("train_acc", self.acc, on_epoch=True, sync_dist=True)

        # add upper bound and lower bound:
        self.log("train_upper_bound", self.upper_bound, on_epoch=True, sync_dist=True)
        self.log("train_lower_bound", self.lower_bound, on_epoch=True, sync_dist=True)

        return self.barlow_loss + self.class_loss + self.lmbd * self.reg_loss

    def sparse_autoencoder(self, z:torch.Tensor) :
        # ----------------encoder--------------
        z = torch.relu(self.auto_encoder(z))
        # ---------top-k sparsity--------------
        z_values, z_indices = torch.topk(z, self.topk, dim=1)

        mask = torch.zeros_like(z)
        z_values_ones = torch.ones_like(z_values)
        mask.scatter_(1, z_indices, z_values_ones)
        sparse_z = z * mask
        # ----------------decoder---------------
        z_out = self.auto_decoder(sparse_z)
        return z_out 
