"""
models.py

Interface chung: forward(x: Tensor[B, T, C]) -> Tensor[B, pred_len, C]

Models gốc:
    TSMixer, RevIN_TSMixer
    DLinear, RevIN_DLinear
    NLinear, RevIN_NLinear
    NBEATS,  RevIN_NBEATS
    NHiTS,   RevIN_NHiTS

Models với STL Decomposition (tiền tố Decomp_):
    Decomp_TSMixer,       Decomp_RevIN_TSMixer
    Decomp_DLinear,       Decomp_RevIN_DLinear
    Decomp_NLinear,       Decomp_RevIN_NLinear
    Decomp_NBEATS,        Decomp_RevIN_NBEATS
    Decomp_NHiTS,         Decomp_RevIN_NHiTS

Kiến trúc Decomp chung:
    x → SeriesDecomposition → trend, seasonal, residual
                                   │                    │
                             TrendBranch         backbone(seasonal + residual)
                           (Linear / MLP)
                                   └──────── + ─────────┘
                                           output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN
from .mixer_layers import TSMixerLayer
from .temporalprojectionlayer import TemporalProjectionLayer
from .decomposition import SeriesDecomposition


# ======================================================================
# TSMixer
# ======================================================================
class TSMixer(nn.Module):
    """TSMixer gốc (Algorithm 1 trong bài báo), không có RevIN wrapper."""

    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float):
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


class RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.revin = RevIN(num_features)
        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.projection(x)
        x = self.revin(x, mode='denorm')
        return x


# ======================================================================
# DLinear – Decomposition Linear (Zeng et al., 2023)
# ======================================================================
class MovingAvg(nn.Module):
    """Moving average để làm mượt chuỗi và lấy trend."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left  = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_pad = F.pad(x.permute(0, 2, 1), (pad_left, pad_right), mode='replicate')
        return self.avg(x_pad).permute(0, 2, 1)


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25):
        super().__init__()
        self.decomp          = MovingAvg(kernel_size)
        self.linear_trend    = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend    = self.decomp(x)
        seasonal = x - trend
        trend_out    = self.linear_trend(trend.permute(0, 2, 1))
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        return (trend_out + seasonal_out).permute(0, 2, 1)


class RevIN_DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25):
        super().__init__()
        self.revin   = RevIN(num_features)
        self.dlinear = DLinear(seq_len, pred_len, num_features, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.revin(x, mode='norm')
        out = self.dlinear(x)
        return self.revin(out, mode='denorm')


# ======================================================================
# NLinear – Normalized Linear (Zeng et al., 2023)
# ======================================================================
class NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last   = x[:, -1:, :]
        x_norm = x - last
        out    = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        return out + last


class RevIN_NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int):
        super().__init__()
        self.revin   = RevIN(num_features)
        self.nlinear = NLinear(seq_len, pred_len, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.revin(x, mode='norm')
        out = self.nlinear(x)
        return self.revin(out, mode='denorm')


# ======================================================================
# N-BEATS (Oreshkin et al., 2020)
# ======================================================================
class NBEATSBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, units: int, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(seq_len, units), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(units, units), nn.ReLU()]
        self.fc_stack      = nn.Sequential(*layers)
        self.backcast_proj = nn.Linear(units, seq_len)
        self.forecast_proj = nn.Linear(units, pred_len)

    def forward(self, x: torch.Tensor):
        h        = self.fc_stack(x)
        backcast = self.backcast_proj(h)
        forecast = self.forecast_proj(h)
        return backcast, forecast


class NBEATS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.pred_len     = pred_len
        self.num_features = num_features
        self.blocks = nn.ModuleList([
            NBEATSBlock(seq_len, pred_len, units, n_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C  = x.shape
        x_flat   = x.permute(0, 2, 1).reshape(B * C, T)
        residual = x_flat
        forecast_sum = torch.zeros(B * C, self.pred_len, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual     = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum.reshape(B, C, self.pred_len).permute(0, 2, 1)


class RevIN_NBEATS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.revin  = RevIN(num_features)
        self.nbeats = NBEATS(seq_len, pred_len, num_features, units, n_blocks, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.revin(x, mode='norm')
        out = self.nbeats(x)
        return self.revin(out, mode='denorm')


# ======================================================================
# N-HiTS (Challu et al., 2023)
# ======================================================================
class NHiTSBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, units: int,
                 pool_size: int = 1, n_layers: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pred_len  = pred_len
        pooled_len     = max(seq_len // pool_size, 1)
        layers = [nn.Linear(pooled_len, units), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(units, units), nn.ReLU()]
        self.fc            = nn.Sequential(*layers)
        self.n_basis       = max(pred_len // pool_size, 1)
        self.backcast_proj = nn.Linear(units, pooled_len)
        self.forecast_proj = nn.Linear(units, self.n_basis)

    def forward(self, x: torch.Tensor):
        if self.pool_size > 1:
            x_pool = F.max_pool1d(
                x.unsqueeze(1), kernel_size=self.pool_size, stride=self.pool_size
            ).squeeze(1)
        else:
            x_pool = x
        h        = self.fc(x_pool)
        backcast = self.backcast_proj(h)
        basis    = self.forecast_proj(h)
        forecast = F.interpolate(
            basis.unsqueeze(1), size=self.pred_len, mode='linear', align_corners=False
        ).squeeze(1)
        backcast_full = F.interpolate(
            backcast.unsqueeze(1), size=x.shape[-1], mode='linear', align_corners=False
        ).squeeze(1)
        return backcast_full, forecast


class NHiTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2):
        super().__init__()
        self.pred_len     = pred_len
        self.num_features = num_features
        if pool_sizes is None:
            pool_sizes = [1, 2, 4]
        self.blocks = nn.ModuleList([
            NHiTSBlock(seq_len, pred_len, units, pool_size=p, n_layers=n_layers)
            for p in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C  = x.shape
        x_flat   = x.permute(0, 2, 1).reshape(B * C, T)
        residual = x_flat
        forecast_sum = torch.zeros(B * C, self.pred_len, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual     = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum.reshape(B, C, self.pred_len).permute(0, 2, 1)


class RevIN_NHiTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2):
        super().__init__()
        self.revin = RevIN(num_features)
        self.nhits = NHiTS(seq_len, pred_len, num_features, units, pool_sizes, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.revin(x, mode='norm')
        out = self.nhits(x)
        return self.revin(out, mode='denorm')


# ======================================================================
# Shared components for Decomp models
# ======================================================================
class TrendBranch(nn.Module):
    """
    Nhánh trend dùng chung cho tất cả Decomp model.

    mode='linear' : Linear(seq_len → pred_len), channel-independent.
                    Ít tham số nhất — phù hợp dữ liệu rất ít, trend smooth.
    mode='mlp'    : Linear → ReLU → Linear, nắm bắt non-linearity nhẹ
                    (ví dụ: tốc độ tăng trưởng thay đổi dần theo thời gian).
    """

    def __init__(self, seq_len: int, pred_len: int,
                 mode: str = 'linear', hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        assert mode in ('linear', 'mlp')
        if mode == 'linear':
            self.net = nn.Linear(seq_len, pred_len)
        else:
            self.net = nn.Sequential(
                nn.Linear(seq_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len),
            )

    def forward(self, trend: torch.Tensor) -> torch.Tensor:
        # trend: [B, T, C] — channel-independent: áp dụng Linear trên chiều T
        return self.net(trend.permute(0, 2, 1)).permute(0, 2, 1)   # [B, pred_len, C]


class DecompWrapper(nn.Module):
    """
    Wrapper decomposition dùng chung cho mọi backbone.

    Pipeline:
        x → SeriesDecomposition → trend, seasonal, residual
                trend → TrendBranch → trend_pred
                seasonal + residual → backbone → season_res_pred
                output = trend_pred + season_res_pred

    Backbone nhận vào tensor [B, T, C] và trả ra [B, pred_len, C].
    Đây là interface chung của tất cả model gốc → không cần sửa backbone.

    Tại sao cộng seasonal + residual trước khi đưa vào backbone:
        - Backbone đủ mạnh để xử lý cả hai thành phần cùng lúc.
        - Tách riêng đòi thêm projection head → tốn tham số, không cần thiết
          với dữ liệu ít điểm như dataset tháng/quý.
        - Residual (noise) thường nhỏ → backbone học cách "bỏ qua" nó tự nhiên.
    """

    def __init__(
        self,
        backbone:        nn.Module,
        seq_len:         int,
        pred_len:        int,
        trend_kernel:    int   = 7,
        seasonal_kernel: int   = 3,
        trend_mode:      str   = 'linear',
        trend_hidden:    int   = 32,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.decomp       = SeriesDecomposition(trend_kernel, seasonal_kernel)
        self.trend_branch = TrendBranch(seq_len, pred_len, trend_mode, trend_hidden, dropout)
        self.backbone     = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, seasonal, residual = self.decomp(x)              # [B, T, C] mỗi phần
        trend_pred      = self.trend_branch(trend)              # [B, pred_len, C]
        season_res_pred = self.backbone(seasonal + residual)    # [B, pred_len, C]
        return trend_pred + season_res_pred                     # [B, pred_len, C]


def _make_decomp(backbone: nn.Module, seq_len: int, pred_len: int,
                 trend_kernel: int, seasonal_kernel: int,
                 trend_mode: str, trend_hidden: int, dropout: float) -> DecompWrapper:
    """Helper nội bộ — tránh lặp kwargs ở mỗi class."""
    return DecompWrapper(
        backbone=backbone,
        seq_len=seq_len,
        pred_len=pred_len,
        trend_kernel=trend_kernel,
        seasonal_kernel=seasonal_kernel,
        trend_mode=trend_mode,
        trend_hidden=trend_hidden,
        dropout=dropout,
    )


# ======================================================================
# Decomp models — mỗi class chỉ khởi tạo backbone rồi bọc DecompWrapper
# Tham số decomp chung: trend_kernel, seasonal_kernel, trend_mode, trend_hidden
# ======================================================================

_DECOMP_DEFAULTS = dict(
    trend_kernel=7,
    seasonal_kernel=3,
    trend_mode='linear',
    trend_hidden=32,
)


class Decomp_TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32):
        super().__init__()
        backbone = TSMixer(seq_len, pred_len, num_features, ff_dim, num_layers, dropout)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32):
        super().__init__()
        backbone = RevIN_TSMixer(seq_len, pred_len, num_features, ff_dim, num_layers, dropout)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_DLinear(nn.Module):
    """
    Ghi chú: DLinear đã có decomposition nội tại (MovingAvg).
    Ở đây thêm STL decomposition bên ngoài → hai tầng tách:
        Tầng ngoài (STL): tách trend dài hạn ra khỏi backbone.
        Tầng trong (DLinear): backbone tự tách seasonal/residual ngắn hạn.
    Kết quả: backbone chỉ xử lý non-trend signal → tập trung hơn.
    """
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = DLinear(seq_len, pred_len, num_features, kernel_size)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_RevIN_DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = RevIN_DLinear(seq_len, pred_len, num_features, kernel_size)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = NLinear(seq_len, pred_len, num_features)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_RevIN_NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = RevIN_NLinear(seq_len, pred_len, num_features)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_NBEATS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = NBEATS(seq_len, pred_len, num_features, units, n_blocks, n_layers)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_RevIN_NBEATS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, n_blocks: int = 3, n_layers: int = 4,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = RevIN_NBEATS(seq_len, pred_len, num_features, units, n_blocks, n_layers)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_NHiTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = NHiTS(seq_len, pred_len, num_features, units, pool_sizes, n_layers)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decomp_RevIN_NHiTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 units: int = 64, pool_sizes: list = None, n_layers: int = 2,
                 trend_kernel: int = 7, seasonal_kernel: int = 3,
                 trend_mode: str = 'linear', trend_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        backbone = RevIN_NHiTS(seq_len, pred_len, num_features, units, pool_sizes, n_layers)
        self.model = _make_decomp(backbone, seq_len, pred_len,
                                  trend_kernel, seasonal_kernel,
                                  trend_mode, trend_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)