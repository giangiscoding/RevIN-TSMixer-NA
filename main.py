import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from train.trainer import train_model, set_seed, create_sequences
from models.revin_tsmixer import RevIN_TSMixer

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang chạy trên thiết bị: {device}")

    csv_path = 'data/data_TSI_v2.csv'
    print(f"Đang tải dữ liệu từ {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file tại {csv_path}.")
        return

    selected_columns = ['Imports', 'IPI', 'DisbursedFDI', 'CompetitorQuantity', 'PromotionAmount', 'Quantity']
    df = df[selected_columns]
    target_col = 'Quantity'
    target_idx = df.columns.get_loc(target_col)
    
    raw_data = df.values.astype(np.float32)
    pred_len = 3
    num_features = raw_data.shape[1] 
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data)
    
    param_grid = {
        'seq_len': [6, 9],
        'n_block': [1, 2],
        'dropout': [0.1, 0.3],
        'batch_size': [2, 4],
        'ff_dim': [64, 128],
        'learning_rate': [1e-4, 1e-5]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Tổng số cấu hình Grid Search: {len(combinations)} tổ hợp\n")

    # BIẾN LƯU KỶ LỤC KỊCH BẢN 1 (MIN MAPE)
    best_s1_params = None
    best_global_val_mape = float('inf')
    best_global_test_mape = float('inf')

    # BIẾN LƯU KỶ LỤC KỊCH BẢN 2 (MIN TOTAL COST)
    best_s2_params = None
    best_global_val_tc = float('inf')
    best_global_test_tc = float('inf')

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Đang thử nghiệm: {params}")
        
        seq_len = params['seq_len']
        X, y = create_sequences(data_scaled, seq_len=seq_len, pred_len=pred_len, target_idx=target_idx)
        
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.1)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=params['batch_size'], shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=params['batch_size'], shuffle=False)
        
        model = RevIN_TSMixer(
            seq_len=params['seq_len'],
            pred_len=pred_len,
            num_features=num_features,
            ff_dim=params['ff_dim'],
            num_layers=params['n_block'],
            dropout=params['dropout']
        )

        try:
            val_mape, test_mape, val_tc, test_tc = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                epochs=50, lr=params['learning_rate'], device=device,
                scaler=scaler, target_idx=target_idx, num_features=num_features
            )
            
            print(f"  -> [S1 - MAPE] Val: {val_mape:.2f}% | Test: {test_mape:.2f}%")
            print(f"  -> [S2 - Cost] Val: {val_tc:,.0f}   | Test: {test_tc:,.0f}")

            # Đánh giá Kịch bản 1
            if val_mape < best_global_val_mape:
                best_global_val_mape = val_mape
                best_global_test_mape = test_mape
                best_s1_params = params
                print(f"S1: KỶ LỤC MAPE MỚI!")

            # Đánh giá Kịch bản 2
            if val_tc < best_global_val_tc:
                best_global_val_tc = val_tc
                best_global_test_tc = test_tc
                best_s2_params = params
                print(f"S2: KỶ LỤC TOTAL COST MỚI!")
                
        except Exception as e:
            print(f"  -> [LỖI] Cấu hình thất bại: {e}")
            continue

    print("\n" + "="*55)
    print("TỔNG KẾT GRID SEARCH THEO 2 KỊCH BẢN TỪ BÀI BÁO")
    print("="*55)
    print("KỊCH BẢN 1: TỐI ƯU HÓA SAI SỐ (MIN MAPE)")
    print(f"Cấu hình tối ưu   : {best_s1_params}")
    print(f"Validation MAPE  : {best_global_val_mape:.2f}%")
    print(f"Test MAPE        : {best_global_test_mape:.2f}%")
    print("-" * 55)
    print("KỊCH BẢN 2: TỐI ƯU HÓA CHI PHÍ (MIN TOTAL COST)")
    print(f"Cấu hình tối ưu   : {best_s2_params}")
    print(f"Validation Cost  : {best_global_val_tc:,.0f}")
    print(f"Test Cost        : {best_global_test_tc:,.0f}")
    print("="*55)

if __name__ == "__main__":
    main()