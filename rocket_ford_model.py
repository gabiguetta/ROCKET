import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class RocketModel(pl.LightningModule):
    """Rocket model for time series classification."""

    def __init__(self,
                 num_kernels=1000,
                 opt_lengths=None,
                 ts_length=500,
                 ridge_lr=1e-3,
                 ridge_thr=0.5):
        super().__init__()

        self.num_kernels = num_kernels
        self.opt_lengths = opt_lengths or [7, 9, 11]
        self.ts_length = ts_length
        self.ridge_lr = ridge_lr
        self.ridge_thr = ridge_thr

        self.kernels = []
        for i in range(num_kernels):
            k_len = np.random.choice(self.opt_lengths)
            A_dil = np.log2((self.ts_length - 1)/(k_len - 1))
            x_dil = np.random.uniform(0, A_dil)
            dilation = int(np.floor(2 ** x_dil))
            padding = np.random.choice([0, ((k_len - 1) * dilation) // 2])
            kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_len, padding=padding, dilation=dilation)
            torch.nn.init.uniform_(kernel.bias, -1., 1.)
            weights = torch.tensor(np.random.normal(0, 1, (1, 1, k_len)))
            kernel.weight.data = weights - torch.mean(weights)
            self.kernels.append(kernel)

    def forward(self, x):
        ppvs = torch.Tensor()
        for kernel in self.kernels:
            k_out = kernel(x)
            ppv = torch.sum(k_out > 0, axis=2)/k_out.shape[-1]
            ppvs = torch.cat((ppvs, ppv), dim=1)
        return ppvs

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
