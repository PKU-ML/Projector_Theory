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

from typing import Any, Dict, List, Sequence
import os
import omegaconf
import torch
import torch.nn as nn
import logging
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
import torch.nn.functional as F
from solo.losses.loss_func import *

def mat_pow(matrix, alpha):
    n = matrix.shape[0]
    res = torch.eye(n).cuda()
    for i in range(alpha):
        res = res @ matrix
    return res

def mat_log(matrix):
    matrix = matrix.to(torch.float32)
    n = matrix.shape[0]
    U, S, V = torch.linalg.svd(matrix, full_matrices=False)
    T = torch.zeros(n).cuda()
    for i in range(n):
        T[i] = torch.log(S[i])
    res = U @ torch.diag(T) @ V
    return res

def renyi_entropy(matrix, alpha):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    
    # Calculate the Renyi entropy
    if alpha == 1:
       entropy = -torch.trace((matrix/n) @ mat_log(matrix/n))
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


class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.final_dim: int = cfg.final_dim
        self.point_num: int = cfg.point_num

        # regularization parameter
        self.lmbd = cfg.lmbd
        # renyi-entropy parameter
        self.alpha = cfg.alpha
        # sparse autoencoder parameter
        self.if_sparse = cfg.sparse_autoencoder # whether use sparse autoencoder or not
        self.topk = cfg.topk
        self.latents_dim = cfg.latents_dim

        # discrete projector
        # if True then employ the sparse autoencoder (can be implemented with other two methods)
        if self.if_sparse :
            self.projector = nn.Sequential()
            self.auto_encoder: nn.Module = nn.Linear(self.features_dim, self.latents_dim)
            self.auto_decoder: nn.Module = nn.Linear(self.latents_dim, self.features_dim)
            self.pre_bias =nn.Parameter(torch.zeros(self.features_dim))
            self.cls = nn.Linear(self.features_dim, 100)
        # if False then just employ the original projector 
        else :
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, self.final_dim),
            )
            self.cls = nn.Linear(self.final_dim, 100)

    
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value




    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        if self.if_sparse :
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()},
                                      {"name": "cls", "params": self.cls.parameters()},
                                      # add learnable params of sparse autoencoder
                                      {"name": "bias", "params": self.pre_bias},                      
                                      {"name": "autoencoder", "params": self.auto_encoder.parameters()},
                                      {"name": "autodecoder", "params": self.auto_decoder.parameters()},
                                     ]
        else :
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()},
                                    {"name": "cls", "params": self.cls.parameters()}]
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

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z1 = out["z1"]
        z2 = self.projector(z1)
        if self.if_sparse :
            # employ the sparse autoencoder as projector
            z2 = self.sparse_autoencoder(z2 - self.pre_bias)
            z2 = z2 + self.pre_bias
        if self.point_num != 0:
            z2 = self.quantize(z2, self.point_num)
        z3 = self.cls(z2)
        out.update({"z2": z2, "z3": z3})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        self.class_loss = out["loss"]
        self.acc = out["acc"]
        logits = torch.cat(out["logits"])
        z1 = F.normalize(torch.cat(out["z1"]))
        z2 = F.normalize(torch.cat(out["z2"]))
        z3 = torch.cat(out["z3"])
        targets = torch.cat(out["targets"])
        pre_acc = (z3.argmax(dim=1)==targets).float().mean()

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        # NCELoss for projector features
        self.nce_loss = simclr_loss_func(
            z2,
            indexes=indexes,
            temperature=self.temperature,
        )

        # NCELoss for encoder features
        self.encoder_nce_loss = simclr_loss_func(
            z1,
            indexes=indexes,
            temperature=self.temperature,
        )

        # online accuracy based on projector features
        self.pre_loss = F.cross_entropy(z3, targets)

        self.reg_loss = matrix_mutual_information(z1, z2, self.alpha)  
        self.lower_bound = -self.encoder_nce_loss - self.reg_loss
        self.upper_bound = renyi_entropy(z1 @ z1.T, self.alpha) - self.reg_loss
        
        # record the vutal params during training
        # online accuracy and loss
        self.log("train_pre_loss", self.pre_loss, on_epoch=True, sync_dist=True)
        self.log("train_pre_acc", pre_acc, on_epoch=True, sync_dist=True)

        # tendency of parameters during training
        self.log("train_nce_loss", self.nce_loss, on_epoch=True, sync_dist=True)
        self.log("I(Z1;R)", -self.encoder_nce_loss, on_epoch=True, sync_dist=True)
        self.log("train_class_loss", self.class_loss, on_epoch=True, sync_dist=True)
        self.log("train_reg_loss", self.reg_loss, on_epoch=True, sync_dist=True)
        
        # record both bounds and online accuracy
        self.log("lower_bound", self.lower_bound, on_epoch=True, sync_dist=True)
        self.log("upper_bound", self.upper_bound, on_epoch=True, sync_dist=True)
        self.log("train_acc", self.acc, on_epoch=True, sync_dist=True)

        return self.nce_loss + self.class_loss + self.lmbd * self.reg_loss

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

        
