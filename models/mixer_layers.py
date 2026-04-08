import torch
import torch.nn as nn

class TSMixerLayer(nn.Module):
    def __init__(self, seq_len: int, num_features: int, ff_dim: int, out_features: int = None, dropout: float = 0.1):
        super(TSMixerLayer, self).__init__()
        self.out_features = out_features if out_features is not None else num_features

        # ================================================================
        # FIX #2: BatchNorm1d phải được khởi tạo với đúng chiều num_features
        # ----------------------------------------------------------------
        # Bài báo mô tả "2D Batch Normalization" (Eq. 2, 7) normalize theo
        # chiều batch (M) và feature (C), giữ nguyên chiều thời gian (T).
        #
        # Trong PyTorch, BatchNorm1d với input 3D [B, C, T] normalize theo
        # chiều C (num_features). Do đó:
        #
        # - Time-Mixing: input lúc normalize là [B, T, C]. Để BN hoạt động
        #   đúng, ta cần reshape về [B*T, C] hoặc dùng chiều C làm
        #   num_features. Ở đây dùng num_features (C) là đúng với bài báo.
        #
        # - Code cũ dùng BatchNorm1d(seq_len) → sai: PyTorch sẽ normalize
        #   theo T thay vì C, không khớp với Eq. 2 của bài báo.
        # ================================================================

        # --- Time Mixing ---
        # BatchNorm1d(num_features) nhận input [B, num_features, T]
        # Trong forward ta sẽ transpose trước khi norm để đúng thứ tự chiều
        self.temporal_norm = nn.BatchNorm1d(num_features)  # FIX: num_features thay vì seq_len
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Feature Mixing ---
        # BatchNorm1d(num_features) nhận input [B, num_features, T]
        # Tương tự: transpose trước khi norm trong forward
        self.feature_norm = nn.BatchNorm1d(num_features)   # Giữ nguyên, đã đúng
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.out_features),
            nn.Dropout(dropout)
        )

        # Xử lý Residual Connection nếu chiều feature bị thay đổi (F < C) theo Eq. 10
        if self.out_features != num_features:
            self.res_projection = nn.Linear(num_features, self.out_features)
        else:
            self.res_projection = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Input x: [B, T, C]

        # ---- Time-Mixing ----
        res_time = x                              # Residual: [B, T, C]

        # FIX #2 (forward): BatchNorm1d yêu cầu input [B, C, T] để normalize đúng theo C.
        # Cũ: temporal_norm(x) với x=[B,T,C] → BN coi T là num_features → SAI
        # Mới: transpose trước → [B, C, T] → BN normalize theo C → đúng Eq. 2
        x_t = x.transpose(1, 2)                  # [B, C, T]
        x_t = self.temporal_norm(x_t)            # BN normalize theo C: [B, C, T]
        x_t = self.temporal_mlp(x_t)             # MLP trên chiều T: [B, C, T]  (Eq. 4)
        x_t = x_t.transpose(1, 2)                # [B, T, C]  (Eq. 5)
        x = x_t + res_time                       # Residual add (Eq. 6)

        # ---- Feature-Mixing ----
        res_feature = self.res_projection(x)     # Chiếu residual nếu F≠C (Eq. 10)

        # Feature norm: input cần [B, C, T] cho BatchNorm1d(num_features)
        # Cũ: transpose → [B,C,T] → feature_norm → đúng (đã đúng trước)
        x_f = x.transpose(1, 2)                  # [B, C, T]
        x_f = self.feature_norm(x_f)             # BN normalize theo C: [B, C, T] (Eq. 7)
        x_f = x_f.transpose(1, 2)                # [B, T, C]
        x_f = self.feature_mlp(x_f)             # MLP trên chiều C: [B, T, F] (Eq. 8)
        x = x_f + res_feature                    # Residual add (Eq. 9 hoặc 10)

        return x