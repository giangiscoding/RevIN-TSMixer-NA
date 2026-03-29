import torch.nn as nn

class TemporalProjectionLayer(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(TemporalProjectionLayer, self).__init__()
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        # Chiếu từ T sang H
        x = self.projection(x)
        # [B, F, H] -> [B, H, F]
        x = x.transpose(1, 2)
        return x