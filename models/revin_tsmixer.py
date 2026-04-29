import torch
import torch.nn as nn
from .revin import RevIN
from .mixer_layers import TSMixerLayer
from .temporalprojectionlayer import TemporalProjectionLayer

class RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int, ff_dim: int, num_layers: int, dropout: float = 0.1):
        super(RevIN_TSMixer, self).__init__()
        self.revin = RevIN(num_features)
    
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