import torch
from torch.utils.data import DataLoader, TensorDataset
from train.trainer import create_sequences

def get_dataloaders(seq_len, batch_size, real_data, pred_len):
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