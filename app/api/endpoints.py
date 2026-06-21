"""
API 路由模块
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional
import os
import tempfile
import time
import numpy as np

from ..schemas import (
    UnmixingRequest,
    UnmixingResponse,
    HealthResponse,
    EndmemberInfo,
    AbundanceMapInfo,
    ErrorResponse,
    DeviceTypeEnum,
    AlterationResponse,
    AlterationEndmemberStats,
)
from src import (
    SpectralUnmixer,
    UnmixingConfig,
    MemoryConfig,
    HSIDataLoader,
    HyperSpectralCube,
)
from src.memory_manager import DeviceType as MemDeviceType


router = APIRouter(prefix="/api/v1", tags=["hyperspectral"])


def get_device_type(device_enum: DeviceTypeEnum) -> MemDeviceType:
    mapping = {
        DeviceTypeEnum.auto: MemDeviceType.AUTO,
        DeviceTypeEnum.cpu: MemDeviceType.CPU,
        DeviceTypeEnum.cuda: MemDeviceType.CUDA,
        DeviceTypeEnum.mps: MemDeviceType.MPS,
    }
    return mapping[device_enum]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    import torch

    cuda_available = torch.cuda.is_available()
    device = "cpu"
    gpu_used = None
    gpu_total = None

    if cuda_available:
        device = "cuda"
        gpu_used = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    elif torch.backends.mps.is_available():
        device = "mps"

    return HealthResponse(
        status="ok",
        device=device,
        cuda_available=cuda_available,
        model_ready=True,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
    )


@router.post("/unmix", response_model=UnmixingResponse)
async def unmix_hyperspectral(
    hdr_file: UploadFile = File(..., description=".hdr 头文件"),
    raw_file: Optional[UploadFile] = File(None, description=".raw 数据文件（可选，若与 hdr 同名可省略）"),
    num_endmembers: int = Form(4, ge=2, le=20, description="端元数量"),
    num_epochs: int = Form(200, ge=10, le=5000, description="训练轮数"),
    learning_rate: float = Form(1e-3, description="学习率"),
    decoder_type: str = Form("linear", description="解码器类型: linear/deconv"),
    normalize: bool = Form(True, description="是否归一化"),
    normalize_method: str = Form("minmax", description="归一化方法"),
    device: str = Form("auto", description="运行设备"),
    enable_mixed_precision: bool = Form(True, description="是否启用混合精度"),
    gpu_memory_fraction: float = Form(0.8, description="GPU显存比例"),
    loss_alpha: float = Form(0.5, description="MSE/SAD损失权重"),
    sparsity_beta: float = Form(0.01, description="稀疏正则化权重"),
    early_stopping_patience: int = Form(20, description="早停耐心值"),
    seed: int = Form(42, description="随机种子"),
    batch_size: Optional[int] = Form(None, description="批大小（自动计算）"),
    verbose: bool = Form(False, description="是否输出详细日志"),
    enable_alteration_analysis: bool = Form(True, description="启用烃类蚀变分析旁路"),
    alteration_enrichment_percentile: float = Form(75.0, description="蚀变富集区分位阈值"),
    alteration_min_ratio: float = Form(0.4, description="2300/1900 深度比最小值"),
):
    """高光谱数据解编接口

    接收 .hdr/.raw 文件，执行无监督光谱解编，返回端元波谱和丰度图
    """
    start_time = time.time()

    hdr_path = None
    raw_path = None
    try:
        if not hdr_file.filename or not hdr_file.filename.lower().endswith(".hdr"):
            raise HTTPException(status_code=400, detail="请上传 .hdr 格式的头文件")

        with tempfile.TemporaryDirectory() as tmpdir:
            hdr_path = os.path.join(tmpdir, hdr_file.filename)
            hdr_content = await hdr_file.read()
            with open(hdr_path, "wb") as f:
                f.write(hdr_content)

            if raw_file is not None:
                raw_filename = raw_file.filename or "data.raw"
                raw_path = os.path.join(tmpdir, raw_filename)
                raw_content = await raw_file.read()
                with open(raw_path, "wb") as f:
                    f.write(raw_content)
            else:
                base, _ = os.path.splitext(hdr_path)
                found = False
                for ext in [".raw", ".dat", ".img"]:
                    candidate = base + ext
                    if os.path.exists(candidate):
                        raw_path = candidate
                        found = True
                        break
                if not found:
                    raise HTTPException(
                        status_code=400,
                        detail="未找到对应的数据文件，请同时上传 .raw 文件"
                    )

            if not os.path.exists(raw_path):
                raise HTTPException(status_code=400, detail="数据文件上传失败")

            unmix_config = UnmixingConfig(
                num_endmembers=num_endmembers,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                loss_alpha=loss_alpha,
                sparsity_beta=sparsity_beta,
                decoder_type=decoder_type,
                normalize=normalize,
                normalize_method=normalize_method,
                early_stopping_patience=early_stopping_patience,
                seed=seed,
                enable_alteration_analysis=enable_alteration_analysis,
                alteration_enrichment_percentile=alteration_enrichment_percentile,
                alteration_min_ratio=alteration_min_ratio,
            )

            dev_type = MemDeviceType.AUTO
            if device == "cpu":
                dev_type = MemDeviceType.CPU
            elif device == "cuda":
                dev_type = MemDeviceType.CUDA
            elif device == "mps":
                dev_type = MemDeviceType.MPS

            memory_config = MemoryConfig(
                device=dev_type,
                gpu_memory_fraction=gpu_memory_fraction,
                enable_mixed_precision=enable_mixed_precision,
            )

            unmixer = SpectralUnmixer(config=unmix_config, memory_config=memory_config)
            result = unmixer.unmix_file(hdr_path, raw_path, verbose=verbose)

            endmember_info_list = []
            for i in range(result.num_endmembers):
                mean_ab = float(np.mean(result.abundance_maps[i, :, :]))
                max_ab = float(np.max(result.abundance_maps[i, :, :]))
                endmember_info_list.append(EndmemberInfo(
                    index=i,
                    name=result.endmember_names[i] if i < len(result.endmember_names) else f"Endmember_{i+1}",
                    mean_abundance=mean_ab,
                    max_abundance=max_ab,
                ))

            abundance_info_list = []
            for i in range(result.num_endmembers):
                ab_map = result.abundance_maps[i, :, :]
                abundance_info_list.append(AbundanceMapInfo(
                    endmember_index=i,
                    endmember_name=result.endmember_names[i] if i < len(result.endmember_names) else f"Endmember_{i+1}",
                    shape=list(ab_map.shape),
                    min_value=float(np.min(ab_map)),
                    max_value=float(np.max(ab_map)),
                    mean_value=float(np.mean(ab_map)),
                ))

            abundance_maps_serializable = []
            for i in range(result.num_endmembers):
                abundance_maps_serializable.append(
                    result.abundance_maps[i, :, :].tolist()
                )

            alteration_resp = AlterationResponse(
                enabled=enable_alteration_analysis
            )
            if result.has_alteration() and result.alteration is not None:
                alt = result.alteration
                em_stats = []
                for i in range(result.num_endmembers):
                    em_stats.append(AlterationEndmemberStats(
                        endmember_index=i,
                        endmember_name=result.endmember_names[i] if i < len(result.endmember_names) else f"Endmember_{i+1}",
                        depth_1900=float(alt.depth_1900[i]),
                        depth_2300=float(alt.depth_2300[i]),
                        ratio_2300_1900=float(alt.ratio_2300_1900[i]),
                        is_carbonate_rich=(i in (alt.endmember_carbonate_rich or [])),
                    ))
                mask_uint8 = alt.get_mask_uint8()
                score = alt.enrichment_score
                alteration_resp = AlterationResponse(
                    enabled=True,
                    success=True,
                    enrichment_threshold=float(alt.enrichment_threshold),
                    enrichment_fraction=float(alt.enrichment_fraction),
                    endmember_stats=em_stats,
                    carbonate_rich_endmembers=list(alt.endmember_carbonate_rich or []),
                    enrichment_mask=mask_uint8.tolist(),
                    enrichment_score=score.tolist() if score is not None else [],
                    enrichment_mask_shape=list(mask_uint8.shape),
                )

            processing_time = time.time() - start_time

            return UnmixingResponse(
                success=True,
                message="光谱解编完成",
                lines=result.lines,
                samples=result.samples,
                bands=result.bands,
                num_endmembers=result.num_endmembers,
                endmembers=result.endmembers.tolist(),
                wavelengths=result.wavelengths.tolist() if result.wavelengths is not None else [],
                abundance_maps=abundance_maps_serializable,
                endmember_info=endmember_info_list,
                abundance_info=abundance_info_list,
                final_loss=result.final_loss,
                loss_history=result.loss_history or [],
                training_epochs=len(result.loss_history) if result.loss_history else 0,
                processing_time=processing_time,
                alteration=alteration_resp,
            )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        detail = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"解编过程出错: {str(e)}"
        )
    finally:
        for path in [hdr_path, raw_path]:
            if path and os.path.exists(path) and os.path.isfile(path):
                try:
                    os.unlink(path)
                except:
                    pass
