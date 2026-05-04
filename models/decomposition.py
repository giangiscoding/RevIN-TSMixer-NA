import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """
    Tách chuỗi thời gian thành 3 thành phần:
        trend      : moving average (low-frequency)
        seasonal   : x - trend  (oscillation around trend)
        residual   : phần còn lại sau khi lấy seasonal trung bình

    Cách tiếp cận:
        - Trend     = AvgPool1d với kernel lớn (window dài, nắm bắt drift chậm)
        - Seasonal  = x - trend  (bao gồm cả seasonality lẫn noise ngắn hạn)
        - Residual  = seasonal - seasonal_smooth
            seasonal_smooth = AvgPool1d với kernel nhỏ hơn (lọc noise ngắn)
            → residual chứa noise/irregular không có cấu trúc rõ
            → seasonal sau khi tách ra sạch hơn

    Lý do dùng hai kernel thay vì một:
        Dữ liệu ít (monthly/quarterly) thường có:
            - Trend chậm → kernel to (~seq_len // 3 hoặc do người dùng chỉ định)
            - Seasonal rõ nhưng ngắn → kernel nhỏ (~3–5) đủ để tách noise
        Tách rõ ba thành phần giúp backbone (TSMixer) chỉ
        phải fit seasonal+residual, giảm gánh nặng đáng kể.
    """

    def __init__(self, trend_kernel: int = 7, seasonal_kernel: int = 3):
        super().__init__()
        # Đảm bảo kernel lẻ để padding symmetric
        self.trend_kernel    = trend_kernel    if trend_kernel    % 2 == 1 else trend_kernel    + 1
        self.seasonal_kernel = seasonal_kernel if seasonal_kernel % 2 == 1 else seasonal_kernel + 1

        self.trend_pool    = nn.AvgPool1d(kernel_size=self.trend_kernel,    stride=1, padding=0)
        self.seasonal_pool = nn.AvgPool1d(kernel_size=self.seasonal_kernel, stride=1, padding=0)

    def _pool(self, x: torch.Tensor, pool: nn.AvgPool1d, kernel: int) -> torch.Tensor:
        """
        Padding replicate hai đầu để giữ nguyên độ dài T sau pooling.
        x: [B, T, C] → output: [B, T, C]
        """
        pad = (kernel - 1) // 2
        # AvgPool1d nhận [B, C, T]
        x_t  = x.permute(0, 2, 1)                                      # [B, C, T]
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')               # [B, C, T+pad*2]
        out  = pool(x_pad)                                              # [B, C, T]
        return out.permute(0, 2, 1)                                     # [B, T, C]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C]
        Returns:
            trend    : [B, T, C]
            seasonal : [B, T, C]  — đã lọc noise thô
            residual : [B, T, C]  — noise/irregular
        """
        trend    = self._pool(x, self.trend_pool,    self.trend_kernel)     # slow drift
        seasonal_raw = x - trend                                             # oscillation + noise
        seasonal_smooth = self._pool(                                        # smooth lại để tách noise
            seasonal_raw, self.seasonal_pool, self.seasonal_kernel
        )
        residual = seasonal_raw - seasonal_smooth                           # noise/irregular
        seasonal = seasonal_smooth

        return trend, seasonal, residual