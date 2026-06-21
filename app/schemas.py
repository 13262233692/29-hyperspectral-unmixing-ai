"""
Pydantic 数据模型 - API 请求/响应结构
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DeviceTypeEnum(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class DecoderTypeEnum(str, Enum):
    linear = "linear"
    deconv = "deconv"


class NormalizeMethodEnum(str, Enum):
    minmax = "minmax"
    mean = "mean"
    zscore = "zscore"


class UnmixingRequest(BaseModel):
    """解编请求参数"""

    num_endmembers: int = Field(default=4, ge=2, le=20, description="端元数量")
    learning_rate: float = Field(default=1e-3, gt=0, description="学习率")
    num_epochs: int = Field(default=200, ge=10, le=5000, description="训练轮数")
    batch_size: Optional[int] = Field(default=None, ge=1, description="批大小（自动计算）")

    decoder_type: DecoderTypeEnum = Field(default=DecoderTypeEnum.linear, description="解码器类型")
    normalize: bool = Field(default=True, description="是否归一化")
    normalize_method: NormalizeMethodEnum = Field(
        default=NormalizeMethodEnum.minmax, description="归一化方法"
    )

    device: DeviceTypeEnum = Field(default=DeviceTypeEnum.auto, description="运行设备")
    enable_mixed_precision: bool = Field(default=True, description="是否启用混合精度")
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="GPU显存使用比例")

    loss_alpha: float = Field(default=0.5, ge=0, le=1, description="MSE/SAD 损失权重")
    sparsity_beta: float = Field(default=0.01, ge=0, description="稀疏正则化权重")

    early_stopping_patience: int = Field(default=20, ge=0, description="早停耐心值")
    seed: int = Field(default=42, description="随机种子")


class EndmemberInfo(BaseModel):
    """端元信息"""

    index: int
    name: str
    mean_abundance: float
    max_abundance: float


class AbundanceMapInfo(BaseModel):
    """丰度图信息"""

    endmember_index: int
    endmember_name: str
    shape: List[int]
    min_value: float
    max_value: float
    mean_value: float


class UnmixingResponse(BaseModel):
    """解编响应结果"""

    success: bool
    message: str = ""

    lines: int = 0
    samples: int = 0
    bands: int = 0
    num_endmembers: int = 0

    endmembers: List[List[float]] = Field(default_factory=list, description="端元波谱矩阵 (n_endmembers, n_bands)")
    wavelengths: List[float] = Field(default_factory=list, description="波长列表")
    abundance_maps: List[List[List[float]]] = Field(default_factory=list, description="丰度图列表")

    endmember_info: List[EndmemberInfo] = Field(default_factory=list)
    abundance_info: List[AbundanceMapInfo] = Field(default_factory=list)

    final_loss: Optional[float] = None
    loss_history: List[float] = Field(default_factory=list)
    training_epochs: int = 0

    processing_time: float = 0.0


class HealthResponse(BaseModel):
    """健康检查响应"""

    model_config = {"protected_namespaces": ()}

    status: str
    device: str
    cuda_available: bool
    model_ready: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


class ErrorResponse(BaseModel):
    """错误响应"""

    success: bool = False
    error: str
    detail: Optional[str] = None
