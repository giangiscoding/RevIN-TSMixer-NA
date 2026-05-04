import torch
import torch.nn as nn


class TSMixerLayer(nn.Module):
    def __init__(self, seq_len: int, num_features: int, ff_dim: int,
                 out_features: int = None, dropout: float = 0.1):
        super(TSMixerLayer, self).__init__()
        self.out_features = out_features if out_features is not None else num_features

        # --- Time Mixing ---
        # BatchNorm1d(num_features) nhận [B, num_features, seq_len]
        self.temporal_norm = nn.BatchNorm1d(num_features)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Feature Mixing ---
        # LayerNorm nhận [B, seq_len, num_features], norm trên chiều cuối (F)
        # Đúng hơn BatchNorm1d vì ta muốn normalize per-timestep, per-sample
        self.feature_norm = nn.LayerNorm(num_features)
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.out_features),
            nn.Dropout(dropout)
        )

        if self.out_features != num_features:
            self.res_projection = nn.Linear(num_features, self.out_features)
        else:
            self.res_projection = nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: [B, T, F]

        # ---- Time Mixing ----
        res_time = x                        # [B, T, F]
        x_t = x.transpose(1, 2)            # [B, F, T]
        x_t = self.temporal_norm(x_t)      # BatchNorm1d(F) trên [B, F, T] → norm đúng chiều F
        x_t = self.temporal_mlp(x_t)       # Linear(T→T) trên chiều cuối T → [B, F, T]
        x_t = x_t.transpose(1, 2)          # [B, T, F]
        x = x_t + res_time                 # residual [B, T, F]

        # ---- Feature Mixing ----
        res_feature = self.res_projection(x)   # [B, T, out_features]
        x_f = self.feature_norm(x)             # LayerNorm trên F → [B, T, F], không cần transpose
        x_f = self.feature_mlp(x_f)            # Linear(F→ff_dim→out_features) trên chiều cuối → [B, T, out_features]
        x = x_f + res_feature                  # residual [B, T, out_features]

        return x