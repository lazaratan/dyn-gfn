import numpy as np
import torch


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())
        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None or self.sigma == -1:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma
        gamma = 1.0 / (1e-8 + 2 * sigma**2)
        K_XY = (-gamma * dnorm2).exp()
        return K_XY
