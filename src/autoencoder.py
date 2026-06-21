"""
1D-CNN 自编码器模型
用于高光谱无监督解编，基于线性光谱混合模型（LSMM）

架构设计：
- 编码器：多层 1D 卷积 + 池化，将输入光谱编码为丰度向量
- 瓶颈层：Softmax 激活，确保丰度和为 1（ASC 约束）
- 解码器：端元矩阵（可学习参数）线性组合，物理可解释
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder1D(nn.Module):
    """1D 卷积编码器
    
    将输入光谱向量编码为丰度潜向量
    输入形状: (batch_size, 1, num_bands)
    输出形状: (batch_size, num_endmembers)
    """

    def __init__(
        self,
        in_bands: int,
        num_endmembers: int = 4,
        hidden_dims: Optional[list] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.num_endmembers = num_endmembers
        self.in_bands = in_bands
        self.kernel_size = kernel_size

        layers = []
        in_channels = 1
        current_length = in_bands

        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_channels = out_channels
            current_length = (current_length + 1) // 2

        self.conv_layers = nn.Sequential(*layers)
        self.flatten_size = in_channels * current_length

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_endmembers),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x


class Decoder1D(nn.Module):
    """1D 解卷积解码器
    
    从丰度向量重建光谱
    采用转置卷积 + 上采样结构，或使用端元矩阵线性组合
    """

    def __init__(
        self,
        out_bands: int,
        num_endmembers: int = 4,
        hidden_dims: Optional[list] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64]

        self.num_endmembers = num_endmembers
        self.out_bands = out_bands
        self.kernel_size = kernel_size

        self.fc_in = nn.Sequential(
            nn.Linear(num_endmembers, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        )

        final_hidden = hidden_dims[-1]
        self.fc_to_conv = nn.Linear(64, final_hidden * 4)
        self.init_length = 4

        layers = []
        in_channels = final_hidden
        current_length = self.init_length

        reversed_hidden = list(reversed(hidden_dims[:-1]))
        for out_channels in reversed_hidden + [1]:
            target_length = min(current_length * 2, out_bands)

            layers.extend([
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                    bias=False,
                ),
            ])

            if out_channels != 1:
                layers.extend([
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ])

            in_channels = out_channels
            current_length = current_length * 2

        self.deconv_layers = nn.Sequential(*layers)

        if current_length != out_bands:
            self.adjust_layer = nn.Conv1d(1, 1, kernel_size=1, stride=1)
            self.target_length = out_bands
        else:
            self.adjust_layer = None
            self.target_length = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(z)
        x = self.fc_to_conv(x)

        batch_size = z.size(0)
        x = x.view(batch_size, -1, self.init_length)

        x = self.deconv_layers(x)

        if self.adjust_layer is not None:
            if x.size(2) != self.target_length:
                x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=True)

        x = x.squeeze(1)
        return x


class LinearDecoder(nn.Module):
    """线性解码器 - 基于端元矩阵的线性光谱混合模型
    
    物理可解释性更强：重建光谱 = 丰度向量 @ 端元矩阵
    端元矩阵作为可学习参数，非负约束（反射率≥0）
    """

    def __init__(self, out_bands: int, num_endmembers: int = 4):
        super().__init__()

        self.num_endmembers = num_endmembers
        self.out_bands = out_bands

        self.endmembers = nn.Parameter(torch.abs(torch.randn(num_endmembers, out_bands) * 0.2 + 0.5))

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        endmembers_pos = torch.relu(self.endmembers)
        reconstructed = torch.matmul(abundances, endmembers_pos)
        return reconstructed

    def get_endmembers(self) -> torch.Tensor:
        return torch.relu(self.endmembers).detach()


class Autoencoder1D(nn.Module):
    """1D-CNN 自编码器 - 光谱解编主模型

    支持两种解码器：
    1. deconv: 转置卷积解码器，重建能力更强
    2. linear: 线性解码器，物理可解释性强，直接输出端元

    丰度约束：
    - 非负约束（ANC）：Softmax 保证非负
    - 和为一约束（ASC）：Softmax 保证和为 1
    """

    def __init__(
        self,
        in_bands: int,
        num_endmembers: int = 4,
        hidden_dims: Optional[list] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        decoder_type: str = "deconv",
    ):
        super().__init__()

        self.in_bands = in_bands
        self.num_endmembers = num_endmembers
        self.decoder_type = decoder_type

        self.encoder = Encoder1D(
            in_bands=in_bands,
            num_endmembers=num_endmembers,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

        if decoder_type == "linear":
            self.decoder = LinearDecoder(
                out_bands=in_bands,
                num_endmembers=num_endmembers,
            )
        elif decoder_type == "deconv":
            self.decoder = Decoder1D(
                out_bands=in_bands,
                num_endmembers=num_endmembers,
                hidden_dims=hidden_dims,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(f"不支持的解码器类型: {decoder_type}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        abundances = F.softmax(z, dim=1)
        return abundances

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abundances = self.encode(x)
        reconstructed = self.decode(abundances)
        return reconstructed, abundances

    def get_endmembers(self) -> Optional[torch.Tensor]:
        if self.decoder_type == "linear":
            return self.decoder.get_endmembers()
        return None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SADLoss(nn.Module):
    """光谱角距离（Spectral Angle Distance）损失
    
    衡量两个光谱向量之间的角度，对光照变化不敏感
    
    数值稳定层：
    1. 死像元/零向量掩码过滤
    2. 双精度 eps 防除零
    3. 双层 cosine 截断（宽松 + 严格）
    4. NaN/Inf 末级防护
    5. acos 边界保护替换策略
    """

    EPS_DIV: float = 1e-10
    EPS_CLAMP: float = 1e-6
    NORM_THRESHOLD: float = 1e-8

    def __init__(self, eps_div: float = 1e-10, eps_clamp: float = 1e-6):
        super().__init__()
        self.EPS_DIV = eps_div
        self.EPS_CLAMP = eps_clamp

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        assert x_pred.shape == x_true.shape, f"形状不匹配: {x_pred.shape} vs {x_true.shape}"

        x_pred_safe = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        x_true_safe = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)

        norm_pred = torch.norm(x_pred_safe, dim=1, p=2)
        norm_true = torch.norm(x_true_safe, dim=1, p=2)

        valid_mask = (norm_pred > self.NORM_THRESHOLD) & (norm_true > self.NORM_THRESHOLD)

        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        dot_product = torch.sum(x_pred_safe * x_true_safe, dim=1)

        denom = norm_pred * norm_true + self.EPS_DIV
        denom = torch.clamp(denom, min=self.EPS_DIV)

        cos_angle = dot_product / denom

        cos_angle = torch.nan_to_num(cos_angle, nan=0.0, posinf=1.0, neginf=-1.0)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.EPS_CLAMP, 1.0 - self.EPS_CLAMP)

        sad = torch.acos(cos_angle)

        sad = torch.nan_to_num(sad, nan=0.0, posinf=0.0, neginf=0.0)

        sad_valid = sad[valid_mask]

        if sad_valid.numel() == 0:
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        return torch.mean(sad_valid)


class SAMLoss(nn.Module):
    """光谱角映射（Spectral Angle Mapper）损失
    
    SAM = (1/π) * arccos(cos(angle))
    归一化到 [0, 1] 区间，便于与 MSE 组合
    完整实现死像元防护与数值稳定层
    """

    EPS_DIV: float = 1e-10
    EPS_CLAMP: float = 1e-6
    NORM_THRESHOLD: float = 1e-8

    def __init__(self, eps_div: float = 1e-10, eps_clamp: float = 1e-6):
        super().__init__()
        self.EPS_DIV = eps_div
        self.EPS_CLAMP = eps_clamp
        self._inv_pi = 1.0 / 3.141592653589793

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        assert x_pred.shape == x_true.shape

        x_pred_safe = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        x_true_safe = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)

        norm_pred = torch.norm(x_pred_safe, dim=1, p=2)
        norm_true = torch.norm(x_true_safe, dim=1, p=2)

        valid_mask = (norm_pred > self.NORM_THRESHOLD) & (norm_true > self.NORM_THRESHOLD)

        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        dot_product = torch.sum(x_pred_safe * x_true_safe, dim=1)

        denom = norm_pred * norm_true + self.EPS_DIV
        denom = torch.clamp(denom, min=self.EPS_DIV)

        cos_angle = dot_product / denom

        cos_angle = torch.nan_to_num(cos_angle, nan=0.0, posinf=1.0, neginf=-1.0)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.EPS_CLAMP, 1.0 - self.EPS_CLAMP)

        sam = self._inv_pi * torch.acos(cos_angle)

        sam = torch.nan_to_num(sam, nan=0.0, posinf=0.0, neginf=0.0)

        sam_valid = sam[valid_mask]
        if sam_valid.numel() == 0:
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        return torch.mean(sam_valid)


class MSE_SAD_Loss(nn.Module):
    """MSE + SAD 组合损失
    
    MSE 关注绝对数值，SAD 关注光谱形状
    
    数值稳定层：
    1. 死像元掩码过滤（零向量不参与 SAD 计算）
    2. NaN/Inf 全链路清除
    3. 末级 NaN 防护返回 0
    """

    def __init__(self, alpha: float = 0.5, eps_div: float = 1e-10, eps_clamp: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="mean")
        self.sad = SADLoss(eps_div=eps_div, eps_clamp=eps_clamp)
        self._norm_threshold = 1e-8

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        x_pred_safe = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        x_true_safe = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)

        norm_true = torch.norm(x_true_safe, dim=1, p=2)
        valid_mask = norm_true > self._norm_threshold

        mse_loss = self.mse(x_pred_safe, x_true_safe)
        mse_loss = torch.nan_to_num(mse_loss, nan=0.0, posinf=1e6, neginf=0.0)

        if torch.any(valid_mask):
            sad_loss = self.sad(x_pred_safe[valid_mask], x_true_safe[valid_mask])
        else:
            sad_loss = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        sad_loss = torch.nan_to_num(sad_loss, nan=0.0, posinf=1e6, neginf=0.0)

        total = self.alpha * mse_loss + (1 - self.alpha) * sad_loss
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)

        return total


class AbundanceSparsityLoss(nn.Module):
    """丰度稀疏性正则化
    
    鼓励丰度向量稀疏，符合实际地质场景（像元通常由少数矿物主导）
    
    数值稳定层：
    1. 算术下溢防护（极小值截断到 min_val）
    2. Softmax 输出的 log 防零（避免 log(0) = -inf）
    3. 死像元零向量过滤
    4. NaN/Inf 末级防护
    """

    def __init__(self, beta: float = 0.01, min_val: float = 1e-12):
        super().__init__()
        self.beta = beta
        self.MIN_VAL = min_val

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        ab = torch.nan_to_num(abundances, nan=0.0, posinf=0.0, neginf=0.0)

        ab_sum = torch.sum(ab, dim=1)
        valid_mask = ab_sum > 1e-8

        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=abundances.device, dtype=abundances.dtype)

        ab_valid = ab[valid_mask]

        ab_clamped = torch.clamp(ab_valid, min=self.MIN_VAL)

        log_ab = torch.log(ab_clamped)
        log_ab = torch.nan_to_num(log_ab, nan=-27.63, posinf=0.0, neginf=-27.63)

        entropy = -torch.sum(ab_valid * log_ab, dim=1)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

        mean_entropy = torch.mean(entropy)
        mean_entropy = torch.nan_to_num(mean_entropy, nan=0.0, posinf=0.0, neginf=0.0)

        return self.beta * mean_entropy


def clip_gradients(model: nn.Module, max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """梯度裁剪阀门
    
    防止梯度爆炸，对 NaN/Inf 梯度进行清零处理
    
    Args:
        model: 待裁剪的模型
        max_norm: 最大范数阈值
        norm_type: 范数类型（2 为 L2 范数）
    
    Returns:
        裁剪前的总范数（用于日志监控）
    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    for p in parameters:
        if p.grad is not None:
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e6, neginf=-1e6)

    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max().item() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        ).item()

    if total_norm > max_norm:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None],
            max_norm=max_norm,
            norm_type=norm_type,
        )

    return total_norm
