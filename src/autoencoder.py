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
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        dot_product = torch.sum(x_pred * x_true, dim=1)
        norm_pred = torch.norm(x_pred, dim=1)
        norm_true = torch.norm(x_true, dim=1)

        cos_angle = dot_product / (norm_pred * norm_true + 1e-8)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        sad = torch.acos(cos_angle)

        return torch.mean(sad)


class MSE_SAD_Loss(nn.Module):
    """MSE + SAD 组合损失
    
    MSE 关注绝对数值，SAD 关注光谱形状
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.sad = SADLoss()

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(x_pred, x_true)
        sad_loss = self.sad(x_pred, x_true)
        return self.alpha * mse_loss + (1 - self.alpha) * sad_loss


class AbundanceSparsityLoss(nn.Module):
    """丰度稀疏性正则化
    
    鼓励丰度向量稀疏，符合实际地质场景（像元通常由少数矿物主导）
    """

    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.beta = beta

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        entropy = -torch.sum(abundances * torch.log(abundances + 1e-8), dim=1)
        return self.beta * torch.mean(entropy)
