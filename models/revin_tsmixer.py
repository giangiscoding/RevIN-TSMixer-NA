import torch
import torch.nn as nn
from .revin import RevIN
from .mixer_layers import TSMixerLayer
from .temporalprojectionlayer import TemporalProjectionLayer

class RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, ff_dim, num_layers, dropout=0.1):
        """
        seq_len: S (VD: 12 tháng lịch sử)
        pred_len: N hoặc H (VD: 3 tháng dự báo)
        num_features: C (Số lượng đặc trưng đầu vào/đầu ra)
        ff_dim: Chiều mở rộng trong Feature-Mixing MLP
        num_layers: Số lượng khối TSMixerLayer muốn xếp chồng
        """
        super(RevIN_TSMixer, self).__init__()
        
        # Khởi tạo RevIN
        self.revin = RevIN(num_features)
        
        # Xếp chồng nhiều lớp TSMixer (thường từ 2 đến 8 lớp tùy độ phức tạp dữ liệu)
        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Khởi tạo lớp chiếu thời gian
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, x):
            # 1. Norm
            x = self.revin(x, mode='norm')
            # 2. TSMixer Blocks
            for layer in self.mixer_layers:
                x = layer(x)
            # 3. Projection
            x = self.projection(x)
            # 4. Denorm
            x = self.revin(x, mode='denorm')
            
            return x