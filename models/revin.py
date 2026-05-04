import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # Lưu trữ statistics
        self.mean = None
        self.stdev = None
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))            
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError("Mode chỉ có thể là 'norm' hoặc 'denorm'")

    def _get_statistics(self, x: torch.Tensor):
        # Tính mean và var dọc theo chiều thời gian (dim=1)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x: torch.Tensor):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denormalize(self, x: torch.Tensor):      
        if self.affine:
            x = (x - self.beta) / (self.gamma + 1e-6)
        x = x * self.stdev + self.mean
        return x