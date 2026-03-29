import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import math
import scipy.stats as stats
from torch.distributions import Normal
from models.inventory_model import Inventory_model

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sequences(data, seq_len, pred_len, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len, target_idx]) 
    return np.array(X), np.array(y)

# 1. HÀM LOSS CHO SCENARIO 1 (MAPE)
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(MAPELoss, self).__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (torch.abs(target) + self.eps))) * 100

# 2. HÀM LOSS CHO SCENARIO 2 (TOTAL COST)
class MinTotalCostLoss(nn.Module):
    def __init__(self, h=2, L=2, o=50000, cs_steps=100):
        super(MinTotalCostLoss, self).__init__()
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

# 4. HÀM TRAIN HOÀN CHỈNH
def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device='gpu', scenario=1,h=2, L=2, o=50000,cs_steps=100):
    criterion = nn.L1Loss() # Hoặc MAPELoss() tùy bạn chọn
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience_limit = 35 if scenario == 1 else 35 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    tc_calculate = Inventory_model(h,L,o,cs_steps)
    model.to(device)
    best_val_score = float('inf') 
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            # 1. Lan truyền xuôi & tính Loss truyền thống
            output = model(batch_x.to(device))
            loss = criterion(output[:, :, 0], batch_y.to(device))
            
            # 2. Cập nhật trọng số mạng nơ-ron
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                v_out = model(vx.to(device))[:, :, 0]
                val_preds.append(v_out)
                val_trues.append(vy.to(device)) 
        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_trues_tensor = torch.cat(val_trues, dim=0)
        val_preds_tensor = torch.clamp(val_preds_tensor, min=0.0)
        
        val_tc = tc_calculate(val_preds_tensor, val_trues_tensor).item() 
        val_preds_flat = val_preds_tensor.flatten()
        val_trues_flat = val_trues_tensor.flatten()
        
        val_mse = torch.mean((val_trues_flat - val_preds_flat)**2).item()
        val_mae = torch.mean(torch.abs(val_trues_flat - val_preds_flat)).item()
        val_rmse = math.sqrt(val_mse)
        val_mape = torch.mean(torch.abs((val_trues_flat - val_preds_flat) / (val_trues_flat + 1e-5))).item() * 100
        
        if scenario == 1:
            current_score = val_mape
        else:
            current_score = val_tc
        
        if current_score < best_val_score:
            best_val_score = current_score
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        scheduler.step(current_score)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | MAPE: {val_mape:.2f}% | Val TC: {val_tc:,.0f}")
        
        if no_improve >= patience_limit:
            print(f"Early Stopping tại epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            t_out = model(tx.to(device))[:, :, 0]
            test_preds.append(t_out)
            test_trues.append(ty.to(device))
    
    test_preds_tensor = torch.clamp(torch.cat(test_preds, dim=0), min=0.0)
    test_trues_tensor = torch.cat(test_trues, dim=0)
    
    test_tc = tc_calculate(test_preds_tensor, test_trues_tensor).item()
    
    test_preds_flat = test_preds_tensor.flatten()
    test_trues_flat = test_trues_tensor.flatten()
    test_mape = torch.mean(torch.abs((test_trues_flat - test_preds_flat) / (test_trues_flat + 1e-5))).item() * 100
    
    print("-" * 30)
    print(f"KẾT QUẢ TRÊN TẬP TEST (Unbiased Evaluation):")
    print(f"Test MAPE: {test_mape:.2f}% | Test TC: {test_tc:,.0f}\n")
    return model