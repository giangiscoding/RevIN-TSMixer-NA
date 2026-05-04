import torch
import torch.nn as nn
from .revin import RevIN
from .mixer_layers import TSMixerLayer
from .temporalprojectionlayer import TemporalProjectionLayer
from .decomposition import SeriesDecomposition


# ======================================================================
# Trend branch — Linear nhẹ (channel-independent, per-feature)
# ======================================================================
class TrendBranch(nn.Module):
    """
    Nhánh trend: dự báo đơn giản vì trend là thành phần smooth, ít phức tạp.

    Hai lựa chọn qua tham số `mode`:
        'linear' : một lớp Linear(seq_len → pred_len) cho mỗi feature,
                   tham số ít nhất, phù hợp dữ liệu rất ít
        'mlp'    : Linear → ReLU → Linear, nắm bắt non-linearity nhẹ trong trend
                   (ví dụ trend thay đổi tốc độ tăng trưởng)
    Channel-independent: mỗi feature xử lý độc lập.
    """

    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 mode: str = 'linear', hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        assert mode in ('linear', 'mlp'), "mode phải là 'linear' hoặc 'mlp'"
        self.mode = mode

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
        """
        Args:
            trend: [B, T, C]
        Returns:
            [B, pred_len, C]
        """
        # Channel-independent: permute để Linear áp dụng trên chiều T
        x = trend.permute(0, 2, 1)          # [B, C, T]
        x = self.net(x)                      # [B, C, pred_len]
        return x.permute(0, 2, 1)           # [B, pred_len, C]


# ======================================================================
# Season+Residual branch — RevIN-TSMixer backbone
# ======================================================================
class SeasonResidualBranch(nn.Module):
    """
    Nhánh seasonal+residual: dùng RevIN-TSMixer vì:
        - Seasonal có phân phối thay đổi → RevIN normalize tốt
        - TSMixer mix cả time và feature → học pattern phức tạp
        - Residual (noise) được backbone học cách "bỏ qua" hoặc smooth

    Input là seasonal + residual (cộng lại trước khi đưa vào backbone)
    để backbone thấy toàn bộ non-trend signal.
    """

    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.revin  = RevIN(num_features)
        self.layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, seasonal: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seasonal: [B, T, C]
            residual: [B, T, C]
        Returns:
            [B, pred_len, C]
        """
        x = seasonal + residual              # gộp hai thành phần non-trend
        x = self.revin(x, mode='norm')
        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)
        x = self.revin(x, mode='denorm')
        return x


# ======================================================================
# DecompRevIN_TSMixer — model chính
# ======================================================================
class DecompRevIN_TSMixer(nn.Module):
    """
    TSMixer với decomposition trước backbone.

    Pipeline:
        x  →  STL decompose  →  trend, seasonal, residual
                                    │                   │
                              TrendBranch        SeasonResidualBranch
                            (Linear / MLP)       (RevIN-TSMixer)
                                    │                   │
                              trend_pred        season_res_pred
                                    └─────── + ─────────┘
                                           output

    Ưu điểm với dữ liệu ít:
        - Backbone chỉ cần fit seasonal+residual → ít overfit hơn
        - Trend được fit bằng model đơn giản → bias thấp, variance thấp
        - Decomposition không có tham số học → không tốn capacity
    """

    def __init__(
        self,
        seq_len:       int,
        pred_len:      int,
        num_features:  int,
        ff_dim:        int   = 32,
        num_layers:    int   = 2,
        dropout:       float = 0.1,
        trend_kernel:  int   = 7,
        seasonal_kernel: int = 3,
        trend_mode:    str   = 'linear',   # 'linear' hoặc 'mlp'
        trend_hidden:  int   = 32,
    ):
        super().__init__()

        self.decomp = SeriesDecomposition(
            trend_kernel=trend_kernel,
            seasonal_kernel=seasonal_kernel,
        )
        self.trend_branch = TrendBranch(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            mode=trend_mode,
            hidden_dim=trend_hidden,
            dropout=dropout,
        )
        self.season_res_branch = SeasonResidualBranch(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, pred_len, C]
        """
        trend, seasonal, residual = self.decomp(x)

        trend_pred      = self.trend_branch(trend)                  # [B, pred_len, C]
        season_res_pred = self.season_res_branch(seasonal, residual) # [B, pred_len, C]

        return trend_pred + season_res_pred                          # [B, pred_len, C]