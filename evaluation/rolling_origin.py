"""
rolling_origin.py – Rolling-origin (walk-forward) evaluation pipeline.

Thay vì single train/val/test split, rolling-origin train lại model trên
expanding window và test trên fold tiếp theo. Điều này:
  1. Tận dụng tối đa 120 điểm dữ liệu nhỏ
  2. Cho kết quả đánh giá ổn định hơn (nhiều test window)
  3. Phản ánh đúng cách deploy thực tế (luôn dùng toàn bộ lịch sử)

Protocol chuẩn:
  - Mỗi fold: train trên [0:origin], val trên [origin:origin+val_size],
    test trên [origin+val_size:origin+val_size+test_size]
  - origin mở rộng mỗi fold thêm step_size điểm
  - Inventory evaluation dùng cùng một Inventory_model cố định

Cost parameters (Section 4.8 bài báo):
  h = 2       : holding cost per unit per month
  L = 2       : lead time (months)
  o = 50,000  : ordering cost per order
  cs ∈ (0,10] : shortage cost (biến, tối ưu hóa)
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import random
import math
from torch.utils.data import DataLoader, TensorDataset
from models.inventory_model import Inventory_model


# ===========================================================================
# Utility
# ===========================================================================

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sequences(data: np.ndarray, seq_len: int, pred_len: int,
                     target_idx: int = 0):
    """Tạo sliding-window sequences từ raw data."""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + pred_len, target_idx])
    return np.array(X), np.array(y)


def compute_metrics(preds: np.ndarray, trues: np.ndarray):
    """
    Tính đầy đủ MSE, MAE, RMSE, MAPE trên numpy arrays.
    preds, trues: [N, pred_len] hoặc [N*pred_len]
    """
    p = preds.flatten()
    t = trues.flatten()
    mse  = np.mean((t - p) ** 2)
    mae  = np.mean(np.abs(t - p))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((t - p) / np.clip(np.abs(t), 1e-5, None))) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ===========================================================================
# Single-fold trainer (dùng chung cho tất cả model)
# ===========================================================================

def train_one_fold(model, train_loader, val_loader, epochs, lr, device,
                   target_idx, patience=30, h=2, L=2, o=50000, cs_steps=100):
    """
    Train model trên một fold, trả về best weights cho S1 và S2.

    Returns:
        best_wts_s1 : state_dict tối ưu theo val MAPE
        best_wts_s2 : state_dict tối ưu theo val Total Cost
        history     : dict chứa val_mape và val_tc qua các epoch
    """
    # ================================================================
    # Ba tầng tách biệt:
    #   1. Train loss  : MSELoss  → gradient mỗi batch
    #   2. Early stop  : val MSE  → dừng khi không cải thiện
    #   3. Best weights: lưu theo val MAPE (S1) và val TC (S2)
    # ================================================================
    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    tc_calc    = Inventory_model(h=h, L=L, o=o, cs_steps=cs_steps).to(device)
    model.to(device)

    # Early stopping theo val MSE
    best_val_mse = float('inf')
    no_improve   = 0

    # Best weights theo từng metric
    best_val_mape = float('inf')
    best_wts_s1   = copy.deepcopy(model.state_dict())

    best_val_tc   = float('inf')
    best_wts_s2   = copy.deepcopy(model.state_dict())

    history = {'val_mape': [], 'val_tc': [], 'val_mse': []}

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            out  = model(bx)[:, :, target_idx]
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validate ----
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                val_preds.append(model(vx.to(device))[:, :, target_idx])
                val_trues.append(vy.to(device))

        vp = torch.clamp(torch.cat(val_preds), min=0.0)
        vt = torch.cat(val_trues)

        # Tầng 2: Early stopping theo val MSE
        val_mse = torch.mean((vt - vp) ** 2).item()
        history['val_mse'].append(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve   = 0
        else:
            no_improve  += 1

        if no_improve >= patience:
            break

        # Tầng 3: Cập nhật best weights theo MAPE và TC
        val_mape = (torch.mean(torch.abs(
            (vt.flatten() - vp.flatten())
            / torch.clamp(vt.flatten(), min=1e-5)
        )).item() * 100)

        val_tc_tensor, _ = tc_calc(vp, vt)
        val_tc = val_tc_tensor.item()

        history['val_mape'].append(val_mape)
        history['val_tc'].append(val_tc)

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_wts_s1   = copy.deepcopy(model.state_dict())

        if val_tc < best_val_tc:
            best_val_tc = val_tc
            best_wts_s2 = copy.deepcopy(model.state_dict())

    return best_wts_s1, best_wts_s2, history, best_val_mape, best_val_tc

# ===========================================================================
# Rolling-origin evaluation
# ===========================================================================

def rolling_origin_evaluate(
    model_fn,           # callable() -> nn.Module  (factory tạo model mới)
    raw_data: np.ndarray,
    seq_len: int,
    pred_len: int,
    target_idx: int,
    lr: float,
    epochs: int,
    device: str,
    batch_size: int     = 4,
    val_size: int       = 11,   # số sequences dùng làm val
    n_folds: int        = 3,    # số fold rolling
    seed: int           = 42,
    h: int              = 2,
    L: int              = 2,
    o: int              = 50000,
    cs_steps: int       = 100,
):
    """
    Rolling-origin (expanding window) evaluation.

    Layout timeline (trên sequences):
    ┌──────────────────────────────────────────────────────┐
    │  Fold 1: [====TRAIN====][=VAL=][TEST]                │
    │  Fold 2: [=======TRAIN=======][=VAL=][TEST]          │
    │  Fold 3: [===========TRAIN===========][=VAL=][TEST]  │
    └──────────────────────────────────────────────────────┘

    Returns:
        results: dict với keys 's1' và 's2', mỗi key chứa:
            - fold_metrics: list of dict (metrics từng fold)
            - mean_metrics: dict (trung bình qua các fold)
            - all_preds   : np.array – toàn bộ predictions ghép lại
            - all_trues   : np.array – toàn bộ ground truth ghép lại
    """
    set_seed(seed)

    X, y = create_sequences(raw_data, seq_len, pred_len, target_idx)
    n_total = len(X)

    # ==================================================================
    # Tính vị trí origin cho từng fold
    # Fold cuối: test kết thúc ở cuối dataset
    # Fold đầu: train tối thiểu 60% sequences
    # ==================================================================
    test_size  = n_total - int(n_total * 0.85)   # ~15% cuối
    # Chia đều khoảng train còn lại thành n_folds fold
    min_train  = int(n_total * 0.60)
    step_size  = max((n_total - test_size - val_size - min_train) // n_folds, 1)

    origins = []
    for k in range(n_folds):
        origin = min_train + k * step_size
        test_end = origin + val_size + test_size
        if test_end > n_total:
            break
        origins.append(origin)

    if len(origins) == 0:
        raise ValueError(
            f"Không đủ data để tạo {n_folds} fold. "
            f"Giảm n_folds hoặc val_size/test_size."
        )

    tc_calc = Inventory_model(h=h, L=L, o=o, cs_steps=cs_steps).to(device)

    # Lưu kết quả từng fold
    fold_results_s1 = []
    fold_results_s2 = []
    all_preds_s1, all_trues_s1 = [], []
    all_preds_s2, all_trues_s2 = [], []

    for fold_idx, origin in enumerate(origins):
        print(f"  Fold {fold_idx+1}/{len(origins)} | "
              f"Train: 0–{origin-1} | "
              f"Val: {origin}–{origin+val_size-1} | "
              f"Test: {origin+val_size}–{origin+val_size+test_size-1}")

        # ---- Tách data ----
        X_train = torch.tensor(X[:origin],                            dtype=torch.float32)
        y_train = torch.tensor(y[:origin],                            dtype=torch.float32)
        X_val   = torch.tensor(X[origin:origin+val_size],             dtype=torch.float32)
        y_val   = torch.tensor(y[origin:origin+val_size],             dtype=torch.float32)
        X_test  = torch.tensor(X[origin+val_size:origin+val_size+test_size], dtype=torch.float32)
        y_test  = torch.tensor(y[origin+val_size:origin+val_size+test_size], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                                  batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                                  batch_size=batch_size, shuffle=False)

        # ---- Khởi tạo model mới cho mỗi fold ----
        set_seed(seed)
        model = model_fn()

        best_wts_s1, best_wts_s2, _, _, _ = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            target_idx=target_idx,
            patience=30,
            h=h, L=L, o=o, cs_steps=cs_steps,
        )

        # ---- Thu thập test trues ----
        test_trues = []
        with torch.no_grad():
            for _, ty in test_loader:
                test_trues.append(ty.to(device))
        tt = torch.cat(test_trues)          # [test_size, pred_len]

        # ---- Đánh giá S1 (best MAPE weights) ----
        model.load_state_dict(best_wts_s1)
        model.eval()
        test_preds_s1 = []
        with torch.no_grad():
            for tx, _ in test_loader:
                test_preds_s1.append(
                    model(tx.to(device))[:, :, target_idx]
                )
        p1 = torch.clamp(torch.cat(test_preds_s1), min=0.0)

        # Forecast metrics
        m1 = compute_metrics(p1.cpu().numpy(), tt.cpu().numpy())
        # Inventory metrics
        tc1, cs1 = tc_calc(p1, tt)
        m1['TC']  = tc1.item()
        m1['cs*'] = cs1
        fold_results_s1.append(m1)
        all_preds_s1.append(p1.cpu().numpy())
        all_trues_s1.append(tt.cpu().numpy())

        # ---- Đánh giá S2 (best TC weights) ----
        model.load_state_dict(best_wts_s2)
        model.eval()
        test_preds_s2 = []
        with torch.no_grad():
            for tx, _ in test_loader:
                test_preds_s2.append(
                    model(tx.to(device))[:, :, target_idx]
                )
        p2 = torch.clamp(torch.cat(test_preds_s2), min=0.0)

        m2 = compute_metrics(p2.cpu().numpy(), tt.cpu().numpy())
        tc2, cs2 = tc_calc(p2, tt)
        m2['TC']  = tc2.item()
        m2['cs*'] = cs2
        fold_results_s2.append(m2)
        all_preds_s2.append(p2.cpu().numpy())
        all_trues_s2.append(tt.cpu().numpy())

        print(f"         S1: MAPE={m1['MAPE']:.2f}%  TC={m1['TC']:,.0f}")
        print(f"         S2: MAPE={m2['MAPE']:.2f}%  TC={m2['TC']:,.0f}")

    # ---- Tổng hợp kết quả qua các fold ----
    def mean_metrics(fold_list):
        keys = fold_list[0].keys()
        return {k: float(np.mean([f[k] for f in fold_list])) for k in keys}

    return {
        's1': {
            'fold_metrics': fold_results_s1,
            'mean_metrics': mean_metrics(fold_results_s1),
            'all_preds':    np.concatenate(all_preds_s1),
            'all_trues':    np.concatenate(all_trues_s1),
        },
        's2': {
            'fold_metrics': fold_results_s2,
            'mean_metrics': mean_metrics(fold_results_s2),
            'all_preds':    np.concatenate(all_preds_s2),
            'all_trues':    np.concatenate(all_trues_s2),
        },
    }