import torch
import torch.nn as nn

class TSMixerLayer(nn.Module):
    def __init__(self, seq_len, num_features, ff_dim, dropout=0.1):
        super(TSMixerLayer, self).__init__()
        # Time Mixing
        self.temporal_norm = nn.BatchNorm1d(seq_len)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Feature Mixing
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Time-Mixing
        res_time = x
        x_t = self.temporal_norm(x)
        x_t = x_t.transpose(1, 2)
        x_t = self.temporal_mlp(x_t)
        x_t = x_t.transpose(1, 2)
        x = x_t + res_time
        
        # Feature-Mixing
        res_feature = x
        x_f = x.transpose(1, 2)
        x_f = self.feature_norm(x_f)
        x_f = x_f.transpose(1, 2)
        x_f = self.feature_mlp(x_f)
        x = x_f + res_feature
        
        return x