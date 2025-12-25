import torch
import numpy as np
from . import flows

# Transforms to be applied to data as preprocessing


class Logit(flows.Flow):
    """Logit mapping of image tensor, see RealNVP paper

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```

    """

    def __init__(self, alpha=0.05):
        """Constructor

        Args:
          alpha: Alpha parameter, see above
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, z):
        beta = 1 - 2 * self.alpha
        sum_dims = list(range(1, z.dim()))
        ls = torch.sum(torch.nn.functional.logsigmoid(z), dim=sum_dims)
        mls = torch.sum(torch.nn.functional.logsigmoid(-z), dim=sum_dims)
        log_det = -np.log(beta) * np.prod([*z.shape[1:]]) + ls + mls
        z = (torch.sigmoid(z) - self.alpha) / beta
        return z, log_det

    def inverse(self, z):
        beta = 1 - 2 * self.alpha
        z = self.alpha + beta * z
        logz = torch.log(z)
        log1mz = torch.log(1 - z)
        z = logz - log1mz
        sum_dims = list(range(1, z.dim()))
        log_det = (
            np.log(beta) * np.prod([*z.shape[1:]])
            - torch.sum(logz, dim=sum_dims)
            - torch.sum(log1mz, dim=sum_dims)
        )
        return z, log_det


class Shift(flows.Flow):
    """Shift data by a fixed constant

    Default is -0.5 to shift data from
    interval [0, 1] to [-0.5, 0.5]
    """

    def __init__(self, shift=-0.5):
        """Constructor

        Args:
          shift: Shift to apply to the data
        """
        super().__init__()
        self.shift = shift

    def forward(self, z):
        z -= self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det

    def inverse(self, z):
        z += self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det
    
class Loft(flows.Flow):
    """Log-transformation for RealNVP

    """
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = z.sign()
        a = z.abs()
        a_M = torch.clamp(a - self.tau, min=0.0)
        a_m = torch.clamp(a, max=self.tau)
        log1p = torch.log1p(a_M)
        z = s * (a_m + log1p)
        sum_dims = list(range(1, z.dim()))
        logdet = -log1p.sum(dim=sum_dims)
        return z, logdet
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = z.sign()
        a = z.abs()
        a_M = torch.clamp(a - self.tau, min=0.0)
        a_m = torch.clamp(a, max=self.tau)
        z = s * (torch.exp(a_M) - 1 + a_m)
        log1p = torch.log1p(a_M)
        sum_dims = list(range(1, z.dim()))
        logdet = log1p.sum(dim=sum_dims)
        return z, logdet
