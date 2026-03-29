import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        num_features (C): Số lượng đặc trưng.
        eps: Tương đương epsilon (\varepsilon) trong bài báo để tránh chia cho 0.
        affine: Cho phép mô hình học các tham số thu phóng (gamma) và dịch chuyển (beta).
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            # Khởi tạo tham số học được (learnable parameters)
            # Ép kích thước (1, 1, C) để PyTorch map chính xác vào chiều Feature cuối cùng
            
            # \gamma_c trong Thuật toán 2 & 3
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features)) 
            
            # \beta_c trong Thuật toán 2 & 3
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError("Mode chỉ có thể là 'norm' hoặc 'denorm'")
        return x

    def _get_statistics(self, x):
        # Đầu vào x: [Batch (M), Time (T), Feature (C)]
        
        # Dòng 4 (Thuật toán 2): Tính \mu_{i,c} dọc theo chiều thời gian (dim=1)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        
        # Dòng 5 (Thuật toán 2): Tính \sigma_{i,c}^2 
        # unbiased=False để tính phương sai chia cho T (chính xác như công thức bài báo)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        # Dòng 7 (Thuật toán 2): \hat{X}_{i,j,c} = (X_{i,j,c} - \mu_{i,c}) / \sigma
        x = (x - self.mean) / self.stdev
        
        if self.affine:
            # Dòng 8 (Thuật toán 2): B_{i,j,c} = \gamma_c * \hat{X}_{i,j,c} + \beta_c
            x = x * self.gamma + self.beta
            
        return x

    def _denormalize(self, x):
        # Đầu vào x lúc này là \hat{B}_{i,j,c} từ Lớp chiếu thời gian
        # Kích thước: [Batch (M), Forecast Length (N), Feature (F)]
        
        if self.affine:
            # Thuật toán 3, Dòng 5 (phần trong ngoặc): (\hat{B}_{i,j,c} - \beta_c) / \gamma_c
            # (Ở đây ta dùng self.beta và self.gamma để sửa luôn lỗi ký hiệu index k của tác giả)
            x = (x - self.beta) / self.gamma
            
        # Thuật toán 3, Dòng 5 (phần ngoài): nhân với \sigma và cộng \mu
        x = x * self.stdev + self.mean
        
        return x