"""
run_baselines.py
================
Pipeline đầy đủ: Optuna TPE search → Rolling-origin evaluation.

Luồng xử lý cho mỗi model × mỗi scenario:
  1. Optuna TPE tìm hyperparameter tốt nhất trên val set (fold đầu tiên)
  2. Dùng hyperparameter đó chạy rolling-origin 3 folds để đánh giá
  3. In bảng so sánh và phân tích nguồn sai số

Tách bạch hai giai đoạn:
  - SEARCH  : dùng fold 0 (train nhỏ nhất) để tìm HP nhanh
  - EVALUATE: dùng rolling-origin 3 folds với HP tốt nhất để đánh giá

Protocol cố định (giống nhau cho TẤT CẢ model):
  pred_len=3, L=2, h=2, o=50000, cs∈(0,10], seed=42
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

from train.trainer import set_seed
from models.revin_tsmixer import RevIN_TSMixer
from models.models import (
    TSMixer, DLinear, NLinear,
    NBEATSBaseline, NHiTSBaseline,
)
from evaluation.rolling_origin import (
    rolling_origin_evaluate, create_sequences,
    train_one_fold, compute_metrics,
)
from models.inventory_model import Inventory_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===========================================================================
# Cấu hình toàn cục – KHÔNG thay đổi khi so sánh các model
# ===========================================================================
CONFIG = {
    'csv_path'   : 'data/data_TSI_v2.csv',
    'features'   : ['Imports', 'IPI', 'DisbursedFDI',
                    'CompetitorQuantity', 'PromotionAmount', 'Quantity'],
    'target'     : 'Quantity',
    'pred_len'   : 3,
    'epochs'     : 50,
    'n_folds'    : 3,
    'val_size'   : 11,
    'seed'       : 42,
    # Inventory (Section 4.8)
    'h'          : 2,
    'L'          : 2,
    'o'          : 50000,
    'cs_steps'   : 100,
    # Optuna
    'n_trials'   : 50,
    'n_startup'  : 10,
}

# ===========================================================================
# Không gian tìm kiếm riêng cho từng model
# Các tham số CHUNG (seq_len, lr, batch_size) áp dụng cho tất cả.
# Tham số ĐẶC THÙ chỉ xuất hiện trong suggest của model đó.
# ===========================================================================

def suggest_common(trial):
    """Hyperparameter chung: seq_len, lr, batch_size, dropout."""
    return {
        'seq_len'      : trial.suggest_categorical('seq_len',    [6, 9, 12]),
        'batch_size'   : trial.suggest_categorical('batch_size', [2, 4, 8]),
        'lr'           : trial.suggest_categorical('lr',         [1e-4, 1e-5]),
        'dropout'      : trial.suggest_float('dropout', 0.0, 0.5, step=0.1),
    }


SEARCH_SPACES = {
    # ── RevIN-TSMixer & TSMixer ──────────────────────────────────────────
    'RevIN-TSMixer': lambda trial: {
        **suggest_common(trial),
        'n_block': trial.suggest_categorical('n_block', [1, 2, 3]),
        'ff_dim' : trial.suggest_categorical('ff_dim',  [32, 64, 128]),
    },
    'TSMixer': lambda trial: {
        **suggest_common(trial),
        'n_block': trial.suggest_categorical('n_block', [1, 2, 3]),
        'ff_dim' : trial.suggest_categorical('ff_dim',  [32, 64, 128]),
    },
    # ── DLinear ─────────────────────────────────────────────────────────
    # DLinear không có dropout hay n_block, chỉ cần seq_len và kernel_size
    'DLinear': lambda trial: {
        'seq_len'     : trial.suggest_categorical('seq_len',     [6, 9, 12]),
        'batch_size'  : trial.suggest_categorical('batch_size',  [2, 4, 8]),
        'lr'          : trial.suggest_categorical('lr',          [1e-4, 1e-5]),
        'dropout'     : 0.0,   # DLinear không dùng dropout
        'kernel_size' : trial.suggest_categorical('kernel_size', [3, 5, 9]),
    },
    # ── NLinear ─────────────────────────────────────────────────────────
    'NLinear': lambda trial: {
        'seq_len'   : trial.suggest_categorical('seq_len',   [6, 9, 12]),
        'batch_size': trial.suggest_categorical('batch_size',[2, 4, 8]),
        'lr'        : trial.suggest_categorical('lr',        [1e-4, 1e-5]),
        'dropout'   : 0.0,
    },
    # ── N-BEATS ─────────────────────────────────────────────────────────
    'N-BEATS': lambda trial: {
        **suggest_common(trial),
        'units'   : trial.suggest_categorical('units',    [32, 64, 128]),
        'n_blocks': trial.suggest_categorical('n_blocks', [2, 3, 4]),
        'n_layers': trial.suggest_categorical('n_layers', [2, 3, 4]),
    },
    # ── N-HiTS ──────────────────────────────────────────────────────────
    'N-HiTS': lambda trial: {
        **suggest_common(trial),
        'units'   : trial.suggest_categorical('units',    [32, 64, 128]),
        'n_layers': trial.suggest_categorical('n_layers', [2, 3]),
        # pool_sizes cố định [1,2,4] – đủ cho seq_len ≥ 6
    },
}


# ===========================================================================
# Factory: tạo model từ hyperparameter dict
# ===========================================================================

def build_model(model_name: str, hp: dict, pred_len: int, num_features: int):
    seq_len = hp['seq_len']

    if model_name == 'RevIN-TSMixer':
        return RevIN_TSMixer(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            ff_dim=hp['ff_dim'], num_layers=hp['n_block'], dropout=hp['dropout'],
        )
    elif model_name == 'TSMixer':
        return TSMixer(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            ff_dim=hp['ff_dim'], num_layers=hp['n_block'], dropout=hp['dropout'],
        )
    elif model_name == 'DLinear':
        kernel = hp.get('kernel_size', 5)
        # kernel phải < seq_len
        kernel = min(kernel, seq_len - 1) if seq_len > 2 else 1
        return DLinear(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            kernel_size=kernel,
        )
    elif model_name == 'NLinear':
        return NLinear(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
        )
    elif model_name == 'N-BEATS':
        return NBEATSBaseline(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            units=hp['units'], n_blocks=hp['n_blocks'], n_layers=hp['n_layers'],
        )
    elif model_name == 'N-HiTS':
        return NHiTSBaseline(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            units=hp['units'], pool_sizes=[1, 2, 4], n_layers=hp['n_layers'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===========================================================================
# Optuna search – dùng fold 0 (train nhỏ nhất) để search nhanh
# ===========================================================================

def optuna_search(
    model_name: str,
    scenario: str,        # 's1' hoặc 's2'
    raw_data: np.ndarray,
    pred_len: int,
    num_features: int,
    target_idx: int,
    device: str,
    n_trials: int = 50,
    seed: int = 42,
):
    """
    Tìm hyperparameter tốt nhất cho model_name × scenario bằng Optuna TPE.

    Dùng fold 0 của rolling-origin (train nhỏ nhất) làm search fold:
      - Train: [0 : min_train]
      - Val  : [min_train : min_train + val_size]
    Điều này đảm bảo search không nhìn thấy test data.

    Returns: best_hp (dict)
    """
    metric_name = 'MAPE' if scenario == 's1' else 'Total Cost'
    print(f"\n  [{model_name}] Optuna TPE - {metric_name} | {n_trials} trials")

    X, y       = create_sequences(raw_data, seq_len=6, pred_len=pred_len,
                                  target_idx=target_idx)
    # Fold 0: min_train = 60% tổng sequences
    n_total    = len(X)
    min_train  = int(n_total * 0.60)
    val_size   = CONFIG['val_size']

    tc_calc = Inventory_model(
        h=CONFIG['h'], L=CONFIG['L'],
        o=CONFIG['o'], cs_steps=CONFIG['cs_steps'],
    ).to(device)

    def objective(trial):
        hp = SEARCH_SPACES[model_name](trial)
        set_seed(seed)

        # Tạo lại sequences với seq_len từ trial
        seq_len = hp['seq_len']
        Xt, yt = create_sequences(raw_data, seq_len=seq_len,
                                  pred_len=pred_len, target_idx=target_idx)
        nt       = len(Xt)
        mt       = int(nt * 0.60)
        val_end  = mt + val_size

        if val_end > nt:
            return float('inf')

        X_train = torch.tensor(Xt[:mt],        dtype=torch.float32)
        y_train = torch.tensor(yt[:mt],        dtype=torch.float32)
        X_val   = torch.tensor(Xt[mt:val_end], dtype=torch.float32)
        y_val   = torch.tensor(yt[mt:val_end], dtype=torch.float32)

        bs = hp['batch_size']
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=bs, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val, y_val),
                                  batch_size=bs, shuffle=False)

        try:
            model = build_model(model_name, hp, pred_len, num_features)
            best_val_mape, best_val_tc = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=CONFIG['epochs'],
                lr=hp['lr'],
                device=device,
                target_idx=target_idx,
                patience=30,
                h=CONFIG['h'], L=CONFIG['L'],
                o=CONFIG['o'], cs_steps=CONFIG['cs_steps'],
            )

            # Đánh giá trên val để lấy metric tối ưu
            wts = best_val_mape if scenario == 's1' else best_val_tc
            model.load_state_dict(wts)
            model.eval()

            val_preds, val_trues = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    val_preds.append(model(vx.to(device))[:, :, target_idx])
                    val_trues.append(vy.to(device))

            vp = torch.clamp(torch.cat(val_preds), min=0.0)
            vt = torch.cat(val_trues)

            if scenario == 's1':
                metric = (torch.mean(torch.abs(
                    (vt.flatten() - vp.flatten())
                    / torch.clamp(vt.flatten(), min=1e-5)
                )).item() * 100)
            else:
                tc_val, _ = tc_calc(vp, vt)
                metric = tc_val.item()

            trial.set_user_attr('hp', hp)
            return metric

        except Exception as e:
            return float('inf')

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=CONFIG['n_startup'],
        multivariate=True,
    )
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f'{model_name}_{scenario}',
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_hp    = best_trial.user_attrs.get('hp', {})
    print(f"         Best val {metric_name}: {best_trial.value:,.2f} | HP: {best_hp}")
    return best_hp


# ===========================================================================
# Đánh giá rolling-origin với HP đã tìm được
# ===========================================================================

def evaluate_with_hp(
    model_name: str,
    hp: dict,
    raw_data: np.ndarray,
    pred_len: int,
    num_features: int,
    target_idx: int,
    device: str,
    seed: int = 42,
):
    """Chạy rolling-origin với hyperparameter cố định."""
    seq_len    = hp['seq_len']
    batch_size = hp['batch_size']
    lr         = hp['lr']

    def model_fn():
        return build_model(model_name, hp, pred_len, num_features)

    return rolling_origin_evaluate(
        model_fn   = model_fn,
        raw_data   = raw_data,
        seq_len    = seq_len,
        pred_len   = pred_len,
        target_idx = target_idx,
        lr         = lr,
        epochs     = CONFIG['epochs'],
        device     = device,
        batch_size = batch_size,
        val_size   = CONFIG['val_size'],
        n_folds    = CONFIG['n_folds'],
        seed       = seed,
        h          = CONFIG['h'],
        L          = CONFIG['L'],
        o          = CONFIG['o'],
        cs_steps   = CONFIG['cs_steps'],
    )


# ===========================================================================
# In kết quả
# ===========================================================================

def print_results_table(all_results: dict, scenario: str):
    s_key    = 's1' if scenario == '1' else 's2'
    title    = "Scenario 1: Min MAPE" if scenario == '1' else "Scenario 2: Min Total Cost"
    sort_key = 'MAPE' if scenario == '1' else 'TC'

    print(f"\n{'='*82}")
    print(f"  {title}  (Rolling-Origin, {CONFIG['n_folds']} folds, mean)")
    print(f"{'='*82}")
    print(f"{'Model':<18} {'MSE':>14} {'MAE':>10} {'RMSE':>10} "
          f"{'MAPE%':>8} {'TC':>12} {'cs*':>6}")
    print(f"{'-'*18} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")

    rows = sorted(
        [(name, res[s_key]['mean_metrics']) for name, res in all_results.items()],
        key=lambda r: r[1][sort_key],
    )
    for name, m in rows:
        print(f"{name:<18} "
              f"{m['MSE']:>14,.0f} "
              f"{m['MAE']:>10,.0f} "
              f"{m['RMSE']:>10,.0f} "
              f"{m['MAPE']:>7.2f}% "
              f"{m['TC']:>12,.0f} "
              f"{m['cs*']:>6.2f}")
    print(f"{'='*82}")


def print_best_hp_table(best_hps: dict):
    """In bảng hyperparameter tốt nhất cho từng model × scenario."""
    print(f"\n{'='*82}")
    print("  HYPERPARAMETER TỐT NHẤT (Optuna TPE)")
    print(f"{'='*82}")
    for model_name, hps in best_hps.items():
        print(f"\n  {model_name}:")
        print(f"    S1 (min MAPE)  : {hps['s1']}")
        print(f"    S2 (min TC)    : {hps['s2']}")
    print(f"{'='*82}")


def analyze_error_source(all_results: dict):
    """Phân tích: sai số nằm ở forecast hay inventory mapping?"""
    print(f"\n{'='*82}")
    print("  PHÂN TÍCH NGUỒN SAI SỐ: FORECAST vs INVENTORY MAPPING")
    print(f"{'='*82}")
    print(f"{'Model':<18} {'S1-MAPE':>8} {'S2-MAPE':>8} "
          f"{'S1-TC':>12} {'S2-TC':>12} {'ΔTC':>10} {'Nhận xét':<25}")
    print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*25}")

    for name, result in all_results.items():
        s1       = result['s1']['mean_metrics']
        s2       = result['s2']['mean_metrics']
        delta_tc = s1['TC'] - s2['TC']

        if delta_tc > 0 and s2['MAPE'] > s1['MAPE']:
            note = "↑MAPE nhưng ↓TC → inv.win"
        elif delta_tc > 0 and s2['MAPE'] <= s1['MAPE']:
            note = "S2 tốt hơn cả hai"
        elif delta_tc <= 0 and s2['MAPE'] <= s1['MAPE']:
            note = "S1 tốt hơn cả hai"
        else:
            note = "S1 tốt hơn TC"

        print(f"{name:<18} "
              f"{s1['MAPE']:>7.2f}% "
              f"{s2['MAPE']:>7.2f}% "
              f"{s1['TC']:>12,.0f} "
              f"{s2['TC']:>12,.0f} "
              f"{delta_tc:>+10,.0f} "
              f"  {note}")

    print(f"\nGhi chú:")
    print(f"  ΔTC > 0 → Tích hợp inventory vào training giúp giảm TC")
    print(f"  ΔTC nhỏ → Forecast accuracy là bottleneck chính, không phải inventory mapping")
    print(f"  ΔTC lớn → Inventory mapping là bottleneck, cải thiện forecast ít hiệu quả")
    print(f"{'='*82}")


def print_fold_detail(all_results: dict, model_name: str):
    result = all_results.get(model_name)
    if result is None:
        return
    print(f"\n--- Chi tiết từng fold: {model_name} ---")
    for s_key, label in [('s1', 'S1-MAPE'), ('s2', 'S2-Cost')]:
        print(f"  {label}:")
        for i, m in enumerate(result[s_key]['fold_metrics']):
            print(f"    Fold {i+1}: MAPE={m['MAPE']:.2f}%  "
                  f"TC={m['TC']:,.0f}  MAE={m['MAE']:,.0f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    set_seed(CONFIG['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device  : {device}")
    print(f"Trials  : {CONFIG['n_trials']} per model per scenario")
    print(f"Folds   : {CONFIG['n_folds']} (rolling-origin)")
    print(f"pred_len: {CONFIG['pred_len']} | h={CONFIG['h']} "
          f"L={CONFIG['L']} o={CONFIG['o']:,}")

    # ---- Load data ----
    df           = pd.read_csv(CONFIG['csv_path'])
    df           = df[CONFIG['features']]
    target_idx   = df.columns.get_loc(CONFIG['target'])
    raw_data     = df.values.astype(np.float32)
    num_features = raw_data.shape[1]
    pred_len     = CONFIG['pred_len']

    print(f"Dataset : {len(raw_data)} điểm | {num_features} features | "
          f"target='{CONFIG['target']}' (idx={target_idx})\n")

    model_names = ['RevIN-TSMixer', 'TSMixer', 'DLinear',
                   'NLinear', 'N-BEATS', 'N-HiTS']

    # ==================================================================
    # Bước 1: Optuna search – tìm HP tốt nhất cho mỗi model × scenario
    # ==================================================================
    print("=" * 60)
    print("BƯỚC 1: OPTUNA TPE SEARCH")
    print("=" * 60)

    best_hps = {}
    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        hp_s1 = optuna_search(
            model_name   = model_name,
            scenario     = 's1',
            raw_data     = raw_data,
            pred_len     = pred_len,
            num_features = num_features,
            target_idx   = target_idx,
            device       = device,
            n_trials     = CONFIG['n_trials'],
            seed         = CONFIG['seed'],
        )
        hp_s2 = optuna_search(
            model_name   = model_name,
            scenario     = 's2',
            raw_data     = raw_data,
            pred_len     = pred_len,
            num_features = num_features,
            target_idx   = target_idx,
            device       = device,
            n_trials     = CONFIG['n_trials'],
            seed         = CONFIG['seed'],
        )
        best_hps[model_name] = {'s1': hp_s1, 's2': hp_s2}

    print_best_hp_table(best_hps)

    # ==================================================================
    # Bước 2: Rolling-origin evaluation với HP tốt nhất
    # Mỗi scenario dùng HP được tối ưu riêng cho scenario đó
    # ==================================================================
    print("\n" + "=" * 60)
    print("BƯỚC 2: ROLLING-ORIGIN EVALUATION")
    print("=" * 60)

    all_results_s1 = {}   # kết quả khi dùng HP tối ưu cho S1
    all_results_s2 = {}   # kết quả khi dùng HP tối ưu cho S2

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")

        # ---- S1: HP tối ưu theo MAPE ----
        print(f"  [S1] HP = {best_hps[model_name]['s1']}")
        res_s1 = evaluate_with_hp(
            model_name   = model_name,
            hp           = best_hps[model_name]['s1'],
            raw_data     = raw_data,
            pred_len     = pred_len,
            num_features = num_features,
            target_idx   = target_idx,
            device       = device,
            seed         = CONFIG['seed'],
        )
        all_results_s1[model_name] = res_s1

        # ---- S2: HP tối ưu theo Total Cost ----
        print(f"  [S2] HP = {best_hps[model_name]['s2']}")
        res_s2 = evaluate_with_hp(
            model_name   = model_name,
            hp           = best_hps[model_name]['s2'],
            raw_data     = raw_data,
            pred_len     = pred_len,
            num_features = num_features,
            target_idx   = target_idx,
            device       = device,
            seed         = CONFIG['seed'],
        )
        all_results_s2[model_name] = res_s2

    # ==================================================================
    # Bước 3: In kết quả
    # Ghép s1 và s2 vào một dict để dùng chung hàm phân tích
    # ==================================================================
    # all_results_combined[model] = {'s1': từ S1-HP, 's2': từ S2-HP}
    all_results_combined = {
        name: {
            's1': all_results_s1[name]['s1'],
            's2': all_results_s2[name]['s2'],
        }
        for name in model_names
    }

    print_results_table(
        {n: {'s1': all_results_s1[n]['s1']} for n in model_names},
        scenario='1',
    )
    print_results_table(
        {n: {'s2': all_results_s2[n]['s2']} for n in model_names},
        scenario='2',
    )
    analyze_error_source(all_results_combined)

    for name in model_names:
        print_fold_detail(
            {'s1': all_results_s1[name], 's2': all_results_s2[name]},
            model_name=name,
        )


if __name__ == '__main__':
    main()