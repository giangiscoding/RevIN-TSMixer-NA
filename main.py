import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from train.trainer import train_model, set_seed, create_sequences
from models.models import (
    TSMixer,       RevIN_TSMixer,
    DLinear,       RevIN_DLinear,
    NLinear,       RevIN_NLinear,
    NBEATS,        RevIN_NBEATS,
    NHiTS,         RevIN_NHiTS,
    Decomp_TSMixer,       Decomp_RevIN_TSMixer,
    Decomp_DLinear,       Decomp_RevIN_DLinear,
    Decomp_NLinear,       Decomp_RevIN_NLinear,
    Decomp_NBEATS,        Decomp_RevIN_NBEATS,
    Decomp_NHiTS,         Decomp_RevIN_NHiTS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ================================================================
# DataLoader
# ================================================================
def get_dataloaders(seq_len, batch_size, real_data, pred_len, target_idx=0):
    X, y = create_sequences(real_data, seq_len, pred_len, target_idx=target_idx)

    train_end = int(len(X) * 0.6)
    val_end   = int(len(X) * 0.8)

    X_train = torch.tensor(X[:train_end],        dtype=torch.float32)
    y_train = torch.tensor(y[:train_end],        dtype=torch.float32)
    X_val   = torch.tensor(X[train_end:val_end], dtype=torch.float32)
    y_val   = torch.tensor(y[train_end:val_end], dtype=torch.float32)
    X_test  = torch.tensor(X[val_end:],          dtype=torch.float32)
    y_test  = torch.tensor(y[val_end:],          dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ================================================================
# Không gian tìm kiếm
# ================================================================
def _suggest_decomp_params(trial):
    return {
        'trend_kernel':    trial.suggest_categorical('trend_kernel',    [3, 5, 7]),
        'seasonal_kernel': trial.suggest_categorical('seasonal_kernel', [3]),
        'trend_mode':      trial.suggest_categorical('trend_mode',      ['linear', 'mlp']),
        'trend_hidden':    trial.suggest_categorical('trend_hidden',    [16, 32]),
    }


def suggest_params_by_model(trial, model_name):
    base_name = model_name.replace('Decomp_', '')
    is_decomp = model_name.startswith('Decomp_')

    params = {}
    params['seq_len']       = trial.suggest_categorical('seq_len',       [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    params['batch_size']    = trial.suggest_categorical('batch_size',    [1, 2, 3, 4])
    params['learning_rate'] = trial.suggest_categorical('learning_rate', [1e-4, 1e-5])

    if base_name in ('TSMixer', 'RevIN_TSMixer'):
        params['n_block'] = trial.suggest_categorical('n_block',  [1, 2, 3])
        params['ff_dim']  = trial.suggest_categorical('ff_dim',   [8, 16, 32, 64, 128])
        params['dropout'] = trial.suggest_categorical('dropout',  [0.1, 0.2, 0.3, 0.4, 0.5])

    elif base_name in ('DLinear', 'RevIN_DLinear'):
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [5, 7, 11, 15, 21, 25])
        params['dropout']     = 0.1

    elif base_name in ('NLinear', 'RevIN_NLinear'):
        params['dropout'] = 0.1

    elif base_name in ('NBEATS', 'RevIN_NBEATS'):
        params['n_blocks'] = trial.suggest_categorical('n_blocks', [1, 2, 3])
        params['units']    = trial.suggest_categorical('units',    [32, 64, 128])
        params['n_layers'] = trial.suggest_categorical('n_layers', [2, 3, 4])
        params['dropout']  = 0.1

    elif base_name in ('NHiTS', 'RevIN_NHiTS'):
        pool_str             = trial.suggest_categorical('pool_sizes', ['1-2-4', '1-2-4-8', '1-2', '2-4-8'])
        params['pool_sizes'] = [int(x) for x in pool_str.split('-')]
        params['units']      = trial.suggest_categorical('units',    [32, 64, 128])
        params['n_layers']   = trial.suggest_categorical('n_layers', [1, 2, 3])
        params['dropout']    = 0.1

    if is_decomp:
        params.update(_suggest_decomp_params(trial))

    return params


# ================================================================
# Build model
# ================================================================
def build_model(model_name, params, pred_len, num_features):
    seq_len   = params['seq_len']
    base_name = model_name.replace('Decomp_', '')
    is_decomp = model_name.startswith('Decomp_')

    decomp_kwargs = dict(
        trend_kernel=params.get('trend_kernel', 7),
        seasonal_kernel=params.get('seasonal_kernel', 3),
        trend_mode=params.get('trend_mode', 'linear'),
        trend_hidden=params.get('trend_hidden', 32),
        dropout=params.get('dropout', 0.1),
    ) if is_decomp else {}

    if base_name in ('TSMixer', 'RevIN_TSMixer'):
        cls_map = {
            'TSMixer':       (TSMixer,       Decomp_TSMixer),
            'RevIN_TSMixer': (RevIN_TSMixer, Decomp_RevIN_TSMixer),
        }
        base_cls, decomp_cls = cls_map[base_name]
        common = dict(seq_len=seq_len, pred_len=pred_len, num_features=num_features,
                      ff_dim=params['ff_dim'], num_layers=params['n_block'],
                      dropout=params['dropout'])
        return decomp_cls(**common, **decomp_kwargs) if is_decomp else base_cls(**common)

    elif base_name in ('DLinear', 'RevIN_DLinear'):
        cls_map = {
            'DLinear':       (DLinear,       Decomp_DLinear),
            'RevIN_DLinear': (RevIN_DLinear, Decomp_RevIN_DLinear),
        }
        base_cls, decomp_cls = cls_map[base_name]
        common = dict(seq_len=seq_len, pred_len=pred_len, num_features=num_features,
                      kernel_size=params.get('kernel_size', 25))
        return decomp_cls(**common, **decomp_kwargs) if is_decomp else base_cls(**common)

    elif base_name in ('NLinear', 'RevIN_NLinear'):
        cls_map = {
            'NLinear':       (NLinear,       Decomp_NLinear),
            'RevIN_NLinear': (RevIN_NLinear, Decomp_RevIN_NLinear),
        }
        base_cls, decomp_cls = cls_map[base_name]
        common = dict(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
        return decomp_cls(**common, **decomp_kwargs) if is_decomp else base_cls(**common)

    elif base_name in ('NBEATS', 'RevIN_NBEATS'):
        cls_map = {
            'NBEATS':       (NBEATS,       Decomp_NBEATS),
            'RevIN_NBEATS': (RevIN_NBEATS, Decomp_RevIN_NBEATS),
        }
        base_cls, decomp_cls = cls_map[base_name]
        common = dict(seq_len=seq_len, pred_len=pred_len, num_features=num_features,
                      units=params.get('units', 64), n_blocks=params.get('n_blocks', 3),
                      n_layers=params.get('n_layers', 4))
        return decomp_cls(**common, **decomp_kwargs) if is_decomp else base_cls(**common)

    elif base_name in ('NHiTS', 'RevIN_NHiTS'):
        pool_sizes = params.get('pool_sizes', [1, 2, 4])
        if any(ps >= seq_len for ps in pool_sizes):
            raise ValueError(f"pool_size {pool_sizes} >= seq_len {seq_len}")
        cls_map = {
            'NHiTS':       (NHiTS,       Decomp_NHiTS),
            'RevIN_NHiTS': (RevIN_NHiTS, Decomp_RevIN_NHiTS),
        }
        base_cls, decomp_cls = cls_map[base_name]
        common = dict(seq_len=seq_len, pred_len=pred_len, num_features=num_features,
                      units=params.get('units', 64), pool_sizes=pool_sizes,
                      n_layers=params.get('n_layers', 2))
        return decomp_cls(**common, **decomp_kwargs) if is_decomp else base_cls(**common)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================================================================
# Objective
# ================================================================
def make_objective(model_name, scenario, raw_data, pred_len, num_features,
                   target_idx, device, seed):
    def objective(trial):
        params = suggest_params_by_model(trial, model_name)
        set_seed(seed)

        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                seq_len=params['seq_len'],
                batch_size=params['batch_size'],
                real_data=raw_data,
                pred_len=pred_len,
                target_idx=target_idx,
            )

            model = build_model(model_name, params, pred_len, num_features)

            val_mape, test_mape_s1, val_tc, test_tc_s2 = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=400,
                lr=params['learning_rate'],
                device=device,
                target_idx=target_idx,
                num_features=num_features,
            )

            trial.set_user_attr('val_mape',     val_mape)
            trial.set_user_attr('test_mape_s1', test_mape_s1)   # test MAPE dùng weights best-MAPE
            trial.set_user_attr('val_tc',       val_tc)
            trial.set_user_attr('test_tc_s2',   test_tc_s2)     # test TC   dùng weights best-TC
            trial.set_user_attr('params',       params)

            return val_mape if scenario == 's1' else val_tc

        except Exception as e:
            print(f"  [Trial {trial.number}] Lỗi: {e}")
            return float('inf')

    return objective


# ================================================================
# Print callback — thêm thông tin decomp
# ================================================================
def _fmt_trial_params(p):
    """Trả về chuỗi tóm tắt hyperparams của trial để in log."""
    parts = [f"seq={p.get('seq_len','?')} lr={p.get('learning_rate', 0):.0e}"]

    if 'ff_dim' in p:
        parts.append(f"ff={p['ff_dim']} block={p.get('n_block','?')} drop={p.get('dropout','?')}")
    if 'kernel_size' in p:
        parts.append(f"kernel={p['kernel_size']}")
    if 'n_blocks' in p and 'ff_dim' not in p:
        parts.append(f"blocks={p['n_blocks']} units={p.get('units')} layers={p.get('n_layers')}")
    if 'pool_sizes' in p:
        parts.append(f"pools={p['pool_sizes']} units={p.get('units')} layers={p.get('n_layers')}")

    # Thông tin decomp (chỉ xuất hiện nếu là Decomp model)
    if 'trend_kernel' in p:
        parts.append(
            f"tk={p['trend_kernel']} sk={p.get('seasonal_kernel','?')} "
            f"tm={p.get('trend_mode','?')} th={p.get('trend_hidden','?')}"
        )

    return ' | '.join(parts)


# ================================================================
# Run study
# ================================================================
def run_study(model_name, scenario, raw_data, pred_len, num_features,
              target_idx, device, seed, n_trials=50):
    scenario_label = "1 - Min MAPE" if scenario == 's1' else "2 - Min Total Cost"
    metric_name    = "MAPE" if scenario == 's1' else "Total Cost"

    print(f"\n{'='*60}")
    print(f"Optuna TPE - {model_name} | Scenario {scenario_label}")
    print(f"Số trials: {n_trials}  |  Startup (random): 10")
    print(f"{'='*60}")

    sampler = TPESampler(seed=seed, n_startup_trials=10, multivariate=True)
    study   = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f'{model_name}_{scenario}',
    )

    objective = make_objective(
        model_name=model_name, scenario=scenario, raw_data=raw_data,
        pred_len=pred_len, num_features=num_features, target_idx=target_idx,
        device=device, seed=seed,
    )

    def print_callback(study, trial):
        if trial.value is None or trial.value == float('inf'):
            return
        p = trial.user_attrs.get('params', {})
        print(f"  Trial {trial.number:3d} | "
              f"Val {metric_name}: {trial.value:>12,.2f} | "
              f"Best: {study.best_value:>12,.2f} | "
              f"{_fmt_trial_params(p)}")

    study.optimize(objective, n_trials=n_trials, callbacks=[print_callback],
                   show_progress_bar=False)

    best_trial      = study.best_trial
    best_params     = best_trial.user_attrs.get('params', {})
    best_val        = best_trial.value
    best_test_mape  = best_trial.user_attrs.get('test_mape_s1', float('nan'))
    best_test_tc    = best_trial.user_attrs.get('test_tc_s2',   float('nan'))

    return best_params, best_val, best_test_mape, best_test_tc, study


# ================================================================
# Print top-5
# ================================================================
def print_top5(study, scenario):
    metric_name = "Val MAPE" if scenario == 's1' else "Val TC"
    test_key    = 'test_mape_s1' if scenario == 's1' else 'test_tc_s2'
    test_label  = "Test MAPE"    if scenario == 's1' else "Test TC"

    trials = sorted(
        [t for t in study.trials if t.value not in (None, float('inf'))],
        key=lambda t: t.value
    )[:5]

    print(f"\nTop 5 trials - Scenario {'1' if scenario == 's1' else '2'}:")
    for t in trials:
        p    = t.user_attrs.get('params', {})
        tval = t.user_attrs.get(test_key, float('nan'))
        fmt  = f"{tval:.2f}%" if scenario == 's1' else f"{tval:,.0f}"
        print(f"  Trial {t.number:3d} | {metric_name}: {t.value:>10,.2f} | "
              f"{test_label}: {fmt} | {_fmt_trial_params(p)}")


# ================================================================
# Main
# ================================================================
def main():
    GLOBAL_SEED = 42
    set_seed(GLOBAL_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang chạy trên thiết bị: {device}")

    csv_path = 'data/data_TSI_v2.csv'
    df       = pd.read_csv(csv_path)

    selected_columns = ['Imports', 'IPI', 'DisbursedFDI',
                        'CompetitorQuantity', 'PromotionAmount', 'Quantity']
    df = df[selected_columns]

    target_idx   = df.columns.get_loc('Quantity')
    raw_data     = df.values.astype(np.float32)
    pred_len     = 3
    num_features = raw_data.shape[1]

    models_to_run = [
        "Decomp_RevIN_NBEATS",
    ]

    for model_name in models_to_run:
        # --- Scenario 1: tối ưu Val MAPE ---
        s1_params, s1_val, s1_test_mape, _, study_s1 = run_study(
            model_name=model_name, scenario='s1',
            raw_data=raw_data, pred_len=pred_len, num_features=num_features,
            target_idx=target_idx, device=device, seed=GLOBAL_SEED, n_trials=50,
        )

        # --- Scenario 2: tối ưu Val Total Cost ---
        s2_params, s2_val, _, s2_test_tc, study_s2 = run_study(
            model_name=model_name, scenario='s2',
            raw_data=raw_data, pred_len=pred_len, num_features=num_features,
            target_idx=target_idx, device=device, seed=GLOBAL_SEED, n_trials=50,
        )

        print("\n" + "=" * 60)
        print(f"KẾT QUẢ CUỐI CÙNG - {model_name}")
        print("=" * 60)
        print("[Scenario 1 – Min MAPE]")
        print(f"  Cấu hình tối ưu : {s1_params}")
        print(f"  Val  MAPE       : {s1_val:.2f}%")
        print(f"  Test MAPE       : {s1_test_mape:.2f}%")
        print("-" * 60)
        print("[Scenario 2 – Min Total Cost]")
        print(f"  Cấu hình tối ưu : {s2_params}")
        print(f"  Val  Cost       : {s2_val:,.0f}")
        print(f"  Test Cost       : {s2_test_tc:,.0f}")
        print("=" * 60)

        print_top5(study_s1, 's1')
        print_top5(study_s2, 's2')


if __name__ == "__main__":
    main()