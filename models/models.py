"""
baselines.py - Các model baseline để so sánh với RevIN-TSMixer.

Mỗi model tuân thủ interface chung:
    forward(x: Tensor[B, T, C]) -> Tensor[B, pred_len, C]

Models:
    TSMixer        - TSMixer gốc không có RevIN (bài báo Section 3.2.2)
    RevIN_TSMixer  - TSMixer có RevIN
    DLinear        - Decomposition Linear (Zeng et al., 2023)
    RevIN_DLinear  - DLinear có RevIN
    NLinear        - Normalized Linear (Zeng et al., 2023)
    RevIN_NLinear  - NLinear có RevIN
    NBEATS - N-BEATS (Oreshkin et al., 2020), generic stack
    RevIN_NBEATS   - N-BEATS có RevIN
    NHiTS  - N-HiTS (Challu et al., 2023)
    RevIN_NHiTS    - N-HiTS có RevIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN
from models.mixer_layers import TSMixerLayer
from models.temporalprojectionlayer import TemporalProjectionLayer


# ======================================================================
# TSMixer
# ======================================================================
class TSMixer(nn.Module):
    """TSMixer gốc (Algorithm 1 trong bài báo), không có RevIN wrapper."""

    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mixer_layers:
            x = layer(x)
        return self.projection(x)

# ======================================================================
# RevIN - TSMixer
# ======================================================================
class RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int, ff_dim: int, num_layers: int, dropout: float = 0.1):
        super(RevIN_TSMixer, self).__init__()
        self.revin = RevIN(num_features)
        
        # Khởi tạo K lớp Mixing (Algorithm 1)
        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, x: torch.Tensor):
        # Đầu vào: [B, T, C]
        x = self.revin(x, mode='norm')       # Chuẩn hóa RevIN
        
        for layer in self.mixer_layers:
            x = layer(x)                     # Qua các lớp Mixer
            
        x = self.projection(x)               # Chiếu thời gian: [B, T, C] -> [B, N, C]
        x = self.revin(x, mode='denorm')     # Giải chuẩn hóa RevIN
        
        return x

# ======================================================================
# 2. DLinear – Decomposition Linear
# Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?"
# Ý tưởng: tách trend (moving average) và remainder, dự báo riêng từng phần
# ======================================================================
class MovingAvg(nn.Module):
    """Moving average để làm mượt chuỗi và lấy trend."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        # Padding hai đầu để giữ nguyên độ dài T
        pad_left  = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_pad = F.pad(x.permute(0, 2, 1),
                      (pad_left, pad_right), mode='replicate')  # [B, C, T+pad]
        trend = self.avg(x_pad).permute(0, 2, 1)               # [B, T, C]
        return trend


class DLinear(nn.Module):
    """
    DLinear: tách x thành trend + residual, dự báo mỗi phần bằng Linear riêng.
    Đơn giản nhưng thường outperform Transformer trên dataset nhỏ.
    """
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25):
        super().__init__()
        self.decomp    = MovingAvg(kernel_size)
        # Mỗi feature có Linear riêng (channel-independent)
        self.linear_trend    = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        trend    = self.decomp(x)           # [B, T, C]
        seasonal = x - trend                # [B, T, C]

        # Linear chiếu trên chiều T: cần [B, C, T] → [B, C, pred_len]
        trend_out    = self.linear_trend(trend.permute(0, 2, 1))     # [B, C, pred_len]
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))

        out = (trend_out + seasonal_out).permute(0, 2, 1)            # [B, pred_len, C]
        return out

class RevIN_DLinear(nn.Module):
    """DLinear bọc RevIN."""
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25):
        super().__init__()
        self.revin = RevIN(num_features)
        self.dlinear = DLinear(seq_len, pred_len, num_features, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        out = self.dlinear(x)
        out = self.revin(out, mode='denorm')
        return out

# ======================================================================
# 3. NLinear – Normalized Linear
# Zeng et al. (2023) – đơn giản hơn DLinear: trừ điểm cuối rồi chiếu
# ======================================================================
class NLinear(nn.Module):
    """
    NLinear: normalize bằng cách trừ giá trị cuối cùng của sequence,
    chiếu bằng Linear, rồi cộng lại. Xử lý distribution shift đơn giản.
    """
    def __init__(self, seq_len: int, pred_len: int, num_features: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        # Lấy điểm cuối làm anchor để normalize
        last = x[:, -1:, :]          # [B, 1, C]
        x_norm = x - last            # [B, T, C]

        # Chiếu trên chiều T
        out = self.linear(x_norm.permute(0, 2, 1))  # [B, C, pred_len]
        out = out.permute(0, 2, 1)                   # [B, pred_len, C]

        # Cộng lại anchor (broadcast qua pred_len)
        out = out + last
        return out

class RevIN_NLinear(nn.Module):
    """NLinear bọc RevIN."""
    def __init__(self, seq_len: int, pred_len: int, num_features: int):
        super().__init__()
        self.revin = RevIN(num_features)
        self.nlinear = NLinear(seq_len, pred_len, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        out = self.nlinear(x)
        out = self.revin(out, mode='denorm')
        return out

# ======================================================================
# 4. N-BEATS – Neural Basis Expansion Analysis
# Oreshkin et al. (2020) – generic stack (không dùng interpretable basis)
# Điều chỉnh: multivariate bằng cách xử lý channel-independent rồi stack
# ======================================================================
class NBEATSBlock(nn.Module):
    """Một block N-BEATS: FC stack → basis expansion → forecast + backcast."""
    def __init__(self, seq_len: int, pred_len: int, units: int,
                 n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(seq_len, units), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(units, units), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)

        self.backcast_proj = nn.Linear(units, seq_len)
        self.forecast_proj = nn.Linear(units, pred_len)

    def forward(self, x: torch.Tensor):
        # x: [B*C, seq_len]
        h         = self.fc_stack(x)
        backcast  = self.backcast_proj(h)   # [B*C, seq_len]
        forecast  = self.forecast_proj(h)   # [B*C, pred_len]
        return backcast, forecast


class NBEATS(nn.Module):
    """
    N-BEATS generic stack - channel-independent (mỗi feature xử lý độc lập).
    Stack nhiều block, residual connection trên backcast.
    """
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.seq_len      = seq_len
        self.pred_len     = pred_len
        self.num_features = num_features

        self.blocks = nn.ModuleList([
            NBEATSBlock(seq_len, pred_len, units, n_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → reshape thành [B*C, T] để xử lý channel-independent
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, T)  # [B*C, T]

        residual = x_flat
        forecast_sum = torch.zeros(B * C, self.pred_len, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual     = residual - backcast   # residual connection
            forecast_sum = forecast_sum + forecast

        # [B*C, pred_len] → [B, pred_len, C]
        out = forecast_sum.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return out

class RevIN_NBEATS(nn.Module):
    """N-BEATS bọc RevIN."""
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.revin = RevIN(num_features)
        self.nbeats = NBEATS(seq_len, pred_len, num_features,
                                     units, n_blocks, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        out = self.nbeats(x)
        out = self.revin(out, mode='denorm')
        return out

# ======================================================================
# 5. N-HiTS – Neural Hierarchical Interpolation for Time Series
# Challu et al. (2023) – multi-rate sampling + hierarchical interpolation
# ======================================================================
class NHiTSBlock(nn.Module):
    """
    Một block N-HiTS với MaxPool downsampling và interpolation upsampling.
    pool_size kiểm soát tỷ lệ downsampling → học pattern ở tần suất khác nhau.
    """
    def __init__(self, seq_len: int, pred_len: int, units: int,
                 pool_size: int = 1, n_layers: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pred_len  = pred_len

        # Sau pooling, độ dài input giảm xuống
        pooled_len = max(seq_len // pool_size, 1)

        layers = [nn.Linear(pooled_len, units), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(units, units), nn.ReLU()]
        self.fc = nn.Sequential(*layers)

        # Số basis = pred_len // pool_size (thô hơn cho block thưa hơn)
        self.n_basis       = max(pred_len // pool_size, 1)
        self.backcast_proj = nn.Linear(units, pooled_len)
        self.forecast_proj = nn.Linear(units, self.n_basis)

    def forward(self, x: torch.Tensor):
        # x: [B*C, seq_len]
        # MaxPool downsampling
        if self.pool_size > 1:
            x_pool = F.max_pool1d(
                x.unsqueeze(1),
                kernel_size=self.pool_size,
                stride=self.pool_size
            ).squeeze(1)                     # [B*C, pooled_len]
        else:
            x_pool = x

        h        = self.fc(x_pool)
        backcast = self.backcast_proj(h)     # [B*C, pooled_len]
        basis    = self.forecast_proj(h)     # [B*C, n_basis]

        # Upsample basis về pred_len bằng linear interpolation
        forecast = F.interpolate(
            basis.unsqueeze(1),
            size=self.pred_len,
            mode='linear',
            align_corners=False
        ).squeeze(1)                         # [B*C, pred_len]

        # Upsample backcast về seq_len
        backcast_full = F.interpolate(
            backcast.unsqueeze(1),
            size=x.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze(1)                         # [B*C, seq_len]

        return backcast_full, forecast


class NHiTS(nn.Module):
    """
    N-HiTS: stack nhiều block với pool_size tăng dần để học multi-scale pattern.
    Channel-independent giống N-BEATS.
    """
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2):
        super().__init__()
        self.seq_len      = seq_len
        self.pred_len     = pred_len
        self.num_features = num_features

        # pool_sizes mặc định: [1, 2, 4] → học pattern từ chi tiết đến thô
        if pool_sizes is None:
            pool_sizes = [1, 2, 4]

        self.blocks = nn.ModuleList([
            NHiTSBlock(seq_len, pred_len, units, pool_size=p, n_layers=n_layers)
            for p in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → [B*C, T]
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, T)

        residual     = x_flat
        forecast_sum = torch.zeros(B * C, self.pred_len, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual     = residual - backcast
            forecast_sum = forecast_sum + forecast

        # [B*C, pred_len] → [B, pred_len, C]
        out = forecast_sum.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return out

class RevIN_NHiTS(nn.Module):
    """N-HiTS bọc RevIN."""
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2):
        super().__init__()
        self.revin = RevIN(num_features)
        self.nhits = NHiTS(seq_len, pred_len, num_features,
                                   units, pool_sizes, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        out = self.nhits(x)
        out = self.revin(out, mode='denorm')
        return out