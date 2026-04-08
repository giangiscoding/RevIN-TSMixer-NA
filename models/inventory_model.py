import torch
import torch.nn as nn
import math
from torch.distributions import Normal

class Inventory_model(nn.Module):
    """
    Probabilistic inventory model – continuous review (r, Q) policy.

    Tham số bài báo (Section 3.3):
        h   : holding cost per unit per month  (hằng số)
        L   : lead time (months)               (hằng số)
        o   : ordering cost per order          (hằng số)
        cs  : shortage cost per unit           (biến, duyệt trên [0.01, 10])

    Các công thức chính:
        Q*      = sqrt(2 * mu_D * o / h)                   (EOQ, dưới Eq. 14)
        alpha   = 1 - (h * Q*) / (cs * mu_D)               (Eq. 14)
        z_alpha = Phi^{-1}(alpha)
        SS      = z_alpha * sigma_D * sqrt(L)               (Eq. 15)
        L(z)    = phi(z) - z*(1 - Phi(z))                  (Eq. 17)
        E(S)    = L(z) * sigma_D * sqrt(L)                  (Eq. 18)
        E(cs)   = cs * E(S) * mu_D / Q*                    (Eq. 19)
        c_o     = (mu_D / Q*) * o                           (Eq. 20)
        c_h     = (Q*/2 + max(SS, 0)) * h                  (Eq. 21)
        TC      = c_o + c_h + E(cs)                        (Eq. 22)

    ================================================================
    FIX #3 – Ý nghĩa của sigma_D
    ----------------------------------------------------------------
    Bài báo ký hiệu sigma_D là "standard deviation of demand" (Table 2).
    Trong lý thuyết tồn kho chuẩn sigma_D đo lường sự biến động của nhu
    cầu thực tế. Tuy nhiên, trong pipeline dự báo–tồn kho, uncertainty
    trong lead time đến từ SAI SỐ DỰ BÁO (forecast error), không phải
    từ nhu cầu thực tế (vì nhu cầu thực chưa biết khi ra quyết định).

    Cách tiếp cận của code (và phù hợp với ý tưởng bài báo Section 3.3):
        residuals = trues - preds
        sigma_D   = std(residuals, dim=1)   ← độ lệch chuẩn của sai số dự báo
                                              theo chiều pred_len cho mỗi mẫu

    Điều này HỢP LÝ vì:
    1. Safety stock mục đích bù cho phần mà mô hình dự báo sai.
    2. Bài báo tính sigma_D từ dữ liệu dự báo/thực theo từng batch (Section
       3.3.1, 3.3.2), không phải từ toàn bộ lịch sử nhu cầu.
    3. Dùng forecast error làm sigma_D là proxy chuẩn trong tài liệu tồn kho
       dự báo (Kourentzes et al., 2020 – Table 3 trong bài báo).

    Ghi chú: sigma_D ở đây được tính theo chiều pred_len (dim=1) cho mỗi
    mẫu trong batch, sau đó average qua các mẫu thông qua torch.mean(best_tc).
    ================================================================
    """

    def __init__(self, h=2, L=2, o=50000, cs_steps=100):
        super(Inventory_model, self).__init__()
        self.h  = h
        self.L  = L
        self.o  = o
        # Duyệt shortage cost cs trên khoảng (0, 10] theo bài báo Section 4.8
        self.cs     = torch.linspace(0.01, 10.0, cs_steps).view(-1, 1)
        self.normal = Normal(0.0, 1.0)

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        """
        Args:
            preds : [N_samples, pred_len]  – dự báo nhu cầu (đã clamp >= 0)
            trues : [N_samples, pred_len]  – nhu cầu thực tế

        Returns:
            mean_best_tc   : scalar – trung bình TC tối ưu qua các mẫu
            mean_best_cs   : float  – giá trị cs* tương ứng (trung bình)
        """

        # ------------------------------------------------------------------
        # 1. Tính sigma_D = std của sai số dự báo theo chiều pred_len
        #    Shape sau: [1, N_samples]  (để broadcast với cs: [cs_steps, 1])
        # ------------------------------------------------------------------
        residuals = trues - preds                                  # [N, pred_len]
        # unbiased=False: tính std theo quần thể (n), nhất quán với RMSE
        sigma_D = torch.std(residuals, dim=1, unbiased=False) + 1e-5  # [N]
        sigma_D = sigma_D.view(1, -1)                              # [1, N]

        # ------------------------------------------------------------------
        # 2. Tính mu_D = trung bình dự báo theo chiều pred_len  (Eq. dưới 14)
        #    mu_D phải > 0 để EOQ có nghĩa
        # ------------------------------------------------------------------
        mu_D = torch.mean(preds, dim=1)                            # [N]
        mu_D = torch.clamp(mu_D, min=1e-4).view(1, -1)            # [1, N]

        cs_device = self.cs.to(preds.device)                       # [cs_steps, 1]

        # ------------------------------------------------------------------
        # 3. Q* – Optimal order quantity (EOQ formula, Section 3.3.1)
        # ------------------------------------------------------------------
        q_star = torch.sqrt((2 * mu_D * self.o) / self.h)         # [1, N]

        # ------------------------------------------------------------------
        # 4. alpha – Service level  (Eq. 14)
        #    alpha = 1 - (h * Q*) / (cs * mu_D)
        #    valid_mask: loại bỏ cs quá thấp khiến alpha <= 0
        # ------------------------------------------------------------------
        alpha_raw  = 1.0 - (self.h * q_star) / (cs_device * mu_D) # [cs_steps, N]
        valid_mask = alpha_raw > 0.001                             # [cs_steps, N]
        alpha      = torch.clamp(alpha_raw, min=1e-4, max=0.9999)  # [cs_steps, N]

        # ------------------------------------------------------------------
        # 5. z_alpha – z-score tương ứng với service level  (dưới Eq. 14)
        # ------------------------------------------------------------------
        z_alpha = self.normal.icdf(alpha)                          # [cs_steps, N]

        # ------------------------------------------------------------------
        # 6. SS – Safety stock  (Eq. 15)
        # ------------------------------------------------------------------
        ss = z_alpha * sigma_D * math.sqrt(self.L)                # [cs_steps, N]

        # ------------------------------------------------------------------
        # 7. L(z) – Loss function  (Eq. 17)
        #    L(z) = phi(z) - z * (1 - Phi(z))
        # ------------------------------------------------------------------
        phi_z = torch.exp(-0.5 * z_alpha ** 2) / math.sqrt(2 * math.pi)
        PHI_z = self.normal.cdf(z_alpha)
        lz    = phi_z - z_alpha * (1.0 - PHI_z)                   # [cs_steps, N]

        # ------------------------------------------------------------------
        # 8. E(S) – Expected shortage per cycle  (Eq. 18)
        # ------------------------------------------------------------------
        e_s = lz * sigma_D * math.sqrt(self.L)                    # [cs_steps, N]

        # ------------------------------------------------------------------
        # 9. Cost components  (Eq. 19–22)
        # ------------------------------------------------------------------
        E_cs = (cs_device * e_s * mu_D) / q_star                  # Eq. 19
        c_o  = (mu_D / q_star) * self.o                           # Eq. 20
        c_h  = (q_star / 2.0 + torch.relu(ss)) * self.h           # Eq. 21
        tc   = c_o + c_h + E_cs                                   # Eq. 22  [cs_steps, N]

        # Vô hiệu hóa các cấu hình cs không hợp lệ
        tc = torch.where(valid_mask, tc, torch.tensor(float('inf'), device=tc.device))

        # ------------------------------------------------------------------
        # 10. Tìm cs* tối ưu (min TC) cho mỗi mẫu
        # ------------------------------------------------------------------
        best_tc, best_idx = torch.min(tc, dim=0)                   # [N]
        best_cs_vals      = cs_device[best_idx].squeeze()          # [N]

        return torch.mean(best_tc), torch.mean(best_cs_vals).item()