import torch
import torch.nn as nn
import numpy as np
import copy
import random
import math
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

def train_model(model, train_loader, val_loader, test_loader,
                epochs, lr, device, target_idx, num_features,
                h=2, L=2, o=50000, cs_steps=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience_limit = 30
    model.to(device)

    tc_calculate = Inventory_model(h=h, L=L, o=o, cs_steps=cs_steps).to(device)

    best_val_mape = float('inf')
    best_wts_s1   = copy.deepcopy(model.state_dict())
    no_improve_s1 = 0

    best_val_tc   = float('inf')
    best_wts_s2   = copy.deepcopy(model.state_dict())
    no_improve_s2 = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)                          # [B, pred_len, C]
            loss   = criterion(output[:, :, target_idx], batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validation ----
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                v_out = model(vx.to(device))[:, :, target_idx]  # [B, pred_len]
                val_preds.append(v_out)
                val_trues.append(vy.to(device))

        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_trues_tensor = torch.cat(val_trues, dim=0)

        # Clamp: dự báo nhu cầu không thể âm
        val_preds_tensor = torch.clamp(val_preds_tensor, min=0.0)

        # Metric 1 – MAPE (Scenario 1)
        val_mape = (
            torch.mean(
                torch.abs(
                    (val_trues_tensor.flatten() - val_preds_tensor.flatten())
                    / torch.clamp(val_trues_tensor.flatten(), min=1e-5)
                )
            ).item() * 100
        )

        # Metric 2 – Total Cost (Scenario 2)
        val_tc_tensor, _ = tc_calculate(val_preds_tensor, val_trues_tensor)
        val_tc = val_tc_tensor.item()

        # Early stopping – Scenario 1 (MAPE)
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_wts_s1   = copy.deepcopy(model.state_dict())
            no_improve_s1 = 0
        else:
            no_improve_s1 += 1

        # Early stopping – Scenario 2 (Total Cost)
        if val_tc < best_val_tc:
            best_val_tc   = val_tc
            best_wts_s2   = copy.deepcopy(model.state_dict())
            no_improve_s2 = 0
        else:
            no_improve_s2 += 1

        # Dừng khi cả hai scenario đều không cải thiện
        if no_improve_s1 >= patience_limit and no_improve_s2 >= patience_limit:
            break

    # ================================================================
    # ĐÁNH GIÁ TẬP TEST
    # ================================================================
    # FIX #1: Xóa vòng lặp dead-code cũ (append tx thay vì model output).
    # test_trues được thu thập sạch từ DataLoader, không lẫn với model output.
    # ----------------------------------------------------------------
    # Thu thập nhãn thực (test_trues) một lần duy nhất
    test_trues = []
    with torch.no_grad():
        for _, ty in test_loader:
            test_trues.append(ty.to(device))
    test_trues_tensor = torch.cat(test_trues, dim=0)   # [N_test, pred_len]

    # ---- Đánh giá Scenario 1: best weights theo MAPE ----
    model.load_state_dict(best_wts_s1)
    model.eval()
    test_preds_s1 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s1.append(model(tx.to(device))[:, :, target_idx])
    p1_tensor = torch.clamp(torch.cat(test_preds_s1, dim=0), min=0.0)

    test_mape_s1 = (
        torch.mean(
            torch.abs(
                (test_trues_tensor.flatten() - p1_tensor.flatten())
                / torch.clamp(test_trues_tensor.flatten(), min=1e-5)
            )
        ).item() * 100
    )

    # ---- Đánh giá Scenario 2: best weights theo Total Cost ----
    model.load_state_dict(best_wts_s2)
    model.eval()
    test_preds_s2 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s2.append(model(tx.to(device))[:, :, target_idx])
    p2_tensor = torch.clamp(torch.cat(test_preds_s2, dim=0), min=0.0)

    test_tc_tensor_s2, _ = tc_calculate(p2_tensor, test_trues_tensor)
    test_tc_s2 = test_tc_tensor_s2.item()

    return best_val_mape, test_mape_s1, best_val_tc, test_tc_s2