import torch
import torch.nn as nn
import numpy as np
import copy
import random
import math
from models.inventory_model import Inventory_model # <-- Gọi trực tiếp class của bạn

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

# Hàm giải chuẩn hóa
def inverse_transform_target(scaled_values, scaler, target_idx, num_features):
    dummy = np.zeros((len(scaled_values), num_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]

def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device, scaler, target_idx, num_features, h=2, L=2, o=50000, cs_steps=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience_limit = 30
    model.to(device)
    
    # Khởi tạo mô hình tồn kho của bạn
    tc_calculate = Inventory_model(h=h, L=L, o=o, cs_steps=cs_steps).to(device)
    
    # Tracking cho Kịch bản 1 (MAPE)
    best_val_mape = float('inf')
    best_wts_s1 = copy.deepcopy(model.state_dict())
    no_improve_s1 = 0
    
    # Tracking cho Kịch bản 2 (Total Cost)
    best_val_tc = float('inf')
    best_wts_s2 = copy.deepcopy(model.state_dict())
    no_improve_s2 = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output[:, :, 0], batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                v_out = model(vx.to(device))[:, :, 0]
                val_preds.append(v_out)
                val_trues.append(vy.to(device)) 
                
        # Lấy shape gốc [số_mẫu, pred_len]
        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_trues_tensor = torch.cat(val_trues, dim=0)
        shape_orig = val_preds_tensor.shape
        
        # Flatten để đưa vào scikit-learn
        val_preds_flat = val_preds_tensor.flatten().cpu().numpy()
        val_trues_flat = val_trues_tensor.flatten().cpu().numpy()
        
        # Giải chuẩn hóa về số lượng thực tế
        val_preds_real_flat = inverse_transform_target(val_preds_flat, scaler, target_idx, num_features)
        val_trues_real_flat = inverse_transform_target(val_trues_flat, scaler, target_idx, num_features)
        val_preds_real_flat = np.clip(val_preds_real_flat, a_min=0.0, a_max=None)
        
        # ---------------------------------------------------------
        # 1. Tính MAPE (Kịch bản 1)
        # ---------------------------------------------------------
        val_mape = np.mean(np.abs((val_trues_real_flat - val_preds_real_flat) / np.clip(val_trues_real_flat, a_min=1e-5, a_max=None))) * 100
        
        # ---------------------------------------------------------
        # 2. Tính Total Cost bằng Inventory_model của bạn (Kịch bản 2)
        # ---------------------------------------------------------
        # Chuyển numpy thành Tensor 2 chiều [số_mẫu, pred_len] và nạp vào tc_calculate
        val_preds_real_tensor = torch.tensor(val_preds_real_flat, dtype=torch.float32, device=device).view(shape_orig)
        val_trues_real_tensor = torch.tensor(val_trues_real_flat, dtype=torch.float32, device=device).view(shape_orig)
        
        val_tc_tensor, val_cs_opt = tc_calculate(val_preds_real_tensor, val_trues_real_tensor)
        val_tc = val_tc_tensor.item()
        
        # Cập nhật Model S1
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_wts_s1 = copy.deepcopy(model.state_dict())
            no_improve_s1 = 0
        else:
            no_improve_s1 += 1
            
        # Cập nhật Model S2
        if val_tc < best_val_tc:
            best_val_tc = val_tc
            best_wts_s2 = copy.deepcopy(model.state_dict())
            no_improve_s2 = 0
        else:
            no_improve_s2 += 1
            
        # Dừng sớm nếu không còn cải thiện ở cả 2 mục tiêu
        if no_improve_s1 >= patience_limit and no_improve_s2 >= patience_limit:
            break

    # ================= ĐÁNH GIÁ TẬP TEST =================
    test_preds, test_trues = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            test_preds.append(tx)
            test_trues.append(ty.to(device))
            
    test_trues_tensor = torch.cat(test_trues, dim=0)
    shape_test = test_trues_tensor.shape
    test_trues_flat = test_trues_tensor.flatten().cpu().numpy()
    test_trues_real_flat = inverse_transform_target(test_trues_flat, scaler, target_idx, num_features)
    test_trues_real_tensor = torch.tensor(test_trues_real_flat, dtype=torch.float32, device=device).view(shape_test)

    # Đánh giá Model tốt nhất theo Kịch bản 1 (MAPE)
    model.load_state_dict(best_wts_s1)
    model.eval()
    test_preds_s1 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s1.append(model(tx.to(device))[:, :, 0])
    p1 = torch.cat(test_preds_s1, dim=0).flatten().cpu().numpy()
    p1_real = np.clip(inverse_transform_target(p1, scaler, target_idx, num_features), a_min=0.0, a_max=None)
    test_mape_s1 = np.mean(np.abs((test_trues_real_flat - p1_real) / np.clip(test_trues_real_flat, a_min=1e-5, a_max=None))) * 100

    # Đánh giá Model tốt nhất theo Kịch bản 2 (Total Cost)
    model.load_state_dict(best_wts_s2)
    model.eval()
    test_preds_s2 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s2.append(model(tx.to(device))[:, :, 0])
            
    p2_tensor = torch.cat(test_preds_s2, dim=0)
    p2_flat = p2_tensor.flatten().cpu().numpy()
    p2_real_flat = np.clip(inverse_transform_target(p2_flat, scaler, target_idx, num_features), a_min=0.0, a_max=None)
    
    # Nạp dạng 2D tensor vào class Inventory_model của bạn
    p2_real_tensor = torch.tensor(p2_real_flat, dtype=torch.float32, device=device).view(shape_test)
    test_tc_tensor_s2, _ = tc_calculate(p2_real_tensor, test_trues_real_tensor)
    test_tc_s2 = test_tc_tensor_s2.item()
    
    return best_val_mape, test_mape_s1, best_val_tc, test_tc_s2