import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from models.revin_tsmixer import RevIN_TSMixer
from training.training import set_seed, train_model, create_sequences


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv('data/data_TSI_v2.csv')

    feature_cols = [
        'Quantity',
        'Imports',
        'IPI',
        'DisbursedFDI',
        'CompetitorQuantity',
        'PromotionAmount'
    ]

    real_data = df[feature_cols].values.astype(np.float32)
    
    num_features = len(feature_cols)
    pred_len = 3
    ff_dim = 128
    dropout = 0.1
    learning_rate = 1e-4
    epochs = 100

    def get_dataloaders(seq_len, batch_size):
        X, y = create_sequences(real_data, seq_len, pred_len, target_idx=0)
        
        train_end = int(len(X) * 0.8)
        val_end = int(len(X) * 0.9)
        
        X_train = torch.tensor(X[:train_end], dtype=torch.float32)
        y_train = torch.tensor(y[:train_end], dtype=torch.float32)
        
        X_val = torch.tensor(X[train_end:val_end], dtype=torch.float32)
        y_val = torch.tensor(y[train_end:val_end], dtype=torch.float32)
        
        X_test = torch.tensor(X[val_end:], dtype=torch.float32)
        y_test = torch.tensor(y[val_end:], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    print("\n" + "-"*50)
    print("MÔ HÌNH 1 - SCENARIO 1 (Loss: MAPE)")
    print("-"*50)
    s1_seq_len, s1_n_block, s1_batch_size = 9, 2, 2
    train_loader_s1, val_loader_s1, test_loader_s1 = get_dataloaders(s1_seq_len, s1_batch_size)
    model_s1 = RevIN_TSMixer(s1_seq_len, pred_len, num_features, ff_dim, s1_n_block, dropout)
    train_model(model_s1, train_loader_s1, val_loader_s1, test_loader_s1, epochs, learning_rate, device, scenario=1,h=2, L=2, o=50000,cs_steps=100)

    print("\n" + "="*50)
    print("MÔ HÌNH 2 - SCENARIO 2 (Loss: Total Inventory Cost)")
    print("="*50)
    s2_seq_len, s2_n_block, s2_batch_size = 9, 2, 2
    train_loader_s2, val_loader_s2, test_loader_s2 = get_dataloaders(s2_seq_len, s2_batch_size)
    
    model_s2 = RevIN_TSMixer(s2_seq_len, pred_len, num_features, ff_dim, s2_n_block, dropout)
    train_model(model_s2, train_loader_s2, val_loader_s2, test_loader_s2, epochs, learning_rate, device, scenario=2,h=2, L=2, o=50000,cs_steps=100)

if __name__ == "__main__":
    main()