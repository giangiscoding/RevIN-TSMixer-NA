import torch
import torch.nn as nn
import math
from torch.distributions import Normal

class Inventory_model(nn.Module):
    def __init__(self, h=2, L=2, o=50000, cs_steps=100):
        super(Inventory_model, self).__init__()
        self.h = h
        self.L = L
        self.o = o
        self.cs = torch.linspace(0.1, 10.0, cs_steps).view(-1, 1)
        self.normal = Normal(0.0, 1.0)

    def forward(self, preds, trues):
        rmse = torch.sqrt(torch.mean((trues - preds)**2, dim=1)) + 1e-5
        rmse = rmse.view(1, -1) 
        mu_D = torch.mean(preds, dim=1)
        mu_D = torch.clamp(mu_D, min=1e-4).view(1, -1)
        cs_device = self.cs.to(preds.device) # Kích thước: [100, 1]
        q_star = torch.sqrt((2 * mu_D * self.o) / self.h)
        
        alpha = 1.0 - (self.h * q_star) / (cs_device * mu_D)
        alpha = torch.clamp(alpha, min=0.5001, max=0.9999) 
        
        z_alpha = self.normal.icdf(alpha)
        ss = torch.relu(z_alpha * rmse * math.sqrt(self.L))
        
        phi_z = torch.exp(-0.5 * z_alpha**2) / math.sqrt(2 * math.pi)
        Phi_z = self.normal.cdf(z_alpha)
        lz = phi_z - z_alpha * (1.0 - Phi_z)
        
        e_s = lz * rmse * math.sqrt(self.L)
        
        E_cs = (cs_device * e_s * mu_D) / q_star
        c_o = (mu_D / q_star) * self.o
        c_h = (q_star / 2.0 + ss) * self.h
        
        tc = c_o + c_h + E_cs
        
        best_tc, _ = torch.min(tc, dim=0)
        
        return torch.mean(best_tc)