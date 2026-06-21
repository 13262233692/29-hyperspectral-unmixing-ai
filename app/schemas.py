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

    enable_alteration_analysis: bool = Field(default=True, description="启用烃类蚀变分析旁路")
    alteration_enrichment_percentile: float = Field(default=75.0, ge=50.0, le=99.0, description="蚀变富集区分位阈值")
    alteration_min_ratio: float = Field(default=0.4, ge=0.0, description="2300/1900 深度比最小值")


class AlterationEndmemberStats(BaseModel):
    """单端元蚀变统计"""

    endmember_index: int
    endmember_name: str
    depth_1900: float
    depth_2300: float
    ratio_2300_1900: float
    is_carbonate_rich: bool


class AlterationResponse(BaseModel):
    """烃类蚀变分析响应"""

    enabled: bool = False
    success: bool = False

    enrichment_threshold: float = 0.0
    enrichment_fraction: float = 0.0

    endmember_stats: List[AlterationEndmemberStats] = Field(default_factory=list)
    carbonate_rich_endmembers: List[int] = Field(default_factory=list)

    enrichment_mask: List[List[int]] = Field(
        default_factory=list, description="油气蚀变富集区二值化掩膜 (lines, samples), 0/255"
    )
    enrichment_score: List[List[float]] = Field(
        default_factory=list, description="归一化蚀变分数热力图 (lines, samples), [0, 1]"
    )
    enrichment_mask_shape: List[int] = Field(default_factory=list)


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

    alteration: AlterationResponse = Field(default_factory=AlterationResponse, description="烃类蚀变分析结果")


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
