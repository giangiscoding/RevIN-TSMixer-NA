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
        self.cs = torch.linspace(0.01, 10.0, cs_steps).view(-1, 1) # Giữ chuẩn 0.1 đến 10
        self.normal = Normal(0.0, 1.0)

    def forward(self, preds, trues):
        rmse = torch.sqrt(torch.mean((trues - preds)**2, dim=1)) + 1e-5
        rmse = rmse.view(1, -1) 
        mu_D = torch.mean(preds, dim=1)
        mu_D = torch.clamp(mu_D, min=1e-4).view(1, -1)
        cs_device = self.cs.to(preds.device)
        
        q_star = torch.sqrt((2 * mu_D * self.o) / self.h)
        
        # 1. Tính alpha gốc chưa qua chỉnh sửa
        alpha_raw = 1.0 - (self.h * q_star) / (cs_device * mu_D)
        
        # 2. TẠO BỨC TƯỜNG CHẶN: Đánh dấu các kịch bản hợp lệ (alpha > 0.001)
        valid_mask = alpha_raw > 0.001
        
        # Vẫn dùng clamp để Pytorch không báo lỗi NaN khi tính z_score phía dưới
        alpha = torch.clamp(alpha_raw, min=1e-4, max=0.9999) 
        
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
        
        # 3. XÂY VÁCH ĐÁ: Nếu alpha không hợp lệ, ép TC bằng VÔ CỰC (Infinity)
        tc = torch.where(valid_mask, tc, torch.tensor(float('inf'), device=tc.device))
        
        # 4. Tối ưu: Lúc này torch.min chắc chắn sẽ chọn đúng đáy chữ U (quanh 1.54)
        best_tc, best_idx = torch.min(tc, dim=0)
        best_cs_vals = cs_device[best_idx].squeeze()
        
        return torch.mean(best_tc), torch.mean(best_cs_vals).item()