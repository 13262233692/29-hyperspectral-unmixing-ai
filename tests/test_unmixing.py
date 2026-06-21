"""
测试脚本 - 验证高光谱解编引擎
生成模拟高光谱数据，执行无监督解编
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    SpectralUnmixer,
    UnmixingConfig,
    MemoryConfig,
    HSIDataLoader,
    HyperSpectralCube,
)
from src.memory_manager import DeviceType


def generate_mock_hsi(
    lines: int = 50,
    samples: int = 50,
    bands: int = 100,
    num_endmembers: int = 4,
    seed: int = 42,
) -> HyperSpectralCube:
    """生成模拟高光谱数据

    使用高斯峰模拟4种矿物端元的吸收特征
    """
    np.random.seed(seed)

    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)

    true_endmembers = np.zeros((num_endmembers, bands), dtype=np.float32)

    peak_centers = [550, 900, 1400, 2200]
    peak_widths = [80, 120, 150, 200]
    base_levels = [0.6, 0.5, 0.7, 0.4]

    for i in range(num_endmembers):
        peak = peak_centers[i % len(peak_centers)]
        width = peak_widths[i % len(peak_widths)]
        base = base_levels[i % len(base_levels)]

        gaussian = np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))
        true_endmembers[i, :] = base - 0.3 * gaussian + 0.05 * np.random.randn(bands)

    true_endmembers = np.clip(true_endmembers, 0, 1)

    abundances = np.random.dirichlet(np.ones(num_endmembers) * 2, size=lines * samples)
    abundances = abundances.astype(np.float32)

    pixel_spectra = abundances @ true_endmembers
    pixel_spectra += 0.02 * np.random.randn(*pixel_spectra.shape).astype(np.float32)
    pixel_spectra = np.clip(pixel_spectra, 0, 1)

    cube_data = pixel_spectra.T.reshape(bands, lines, samples)

    return HyperSpectralCube(
        data=cube_data,
        wavelengths=wavelengths,
        lines=lines,
        samples=samples,
        bands=bands,
        interleave="bsq",
        dtype="float32",
    ), true_endmembers, abundances


def test_basic_unmixing():
    """基础解编功能测试"""
    print("=" * 60)
    print("测试 1: 基础无监督解编")
    print("=" * 60)

    cube, true_endmembers, true_abundances = generate_mock_hsi(
        lines=30, samples=30, bands=80, num_endmembers=4, seed=42
    )

    print(f"数据形状: {cube.shape} (bands, lines, samples)")
    print(f"像素总数: {cube.lines * cube.samples}")

    unmix_config = UnmixingConfig(
        num_endmembers=4,
        num_epochs=300,
        learning_rate=5e-4,
        decoder_type="linear",
        normalize=True,
        normalize_method="minmax",
        loss_alpha=0.7,
        sparsity_beta=0.001,
        early_stopping_patience=50,
        seed=42,
    )

    memory_config = MemoryConfig(
        device=DeviceType.CPU,
        enable_mixed_precision=False,
    )

    unmixer = SpectralUnmixer(config=unmix_config, memory_config=memory_config)
    result = unmixer.unmix(cube, verbose=True)

    print(f"\n解编完成!")
    print(f"端元数量: {result.num_endmembers}")
    print(f"丰度图形状: {result.abundance_maps.shape}")
    print(f"端元矩阵形状: {result.endmembers.shape}")
    print(f"最终损失: {result.final_loss:.6f}")
    print(f"训练轮数: {len(result.loss_history)}")

    print("\n端元平均丰度:")
    for i in range(result.num_endmembers):
        mean_ab = np.mean(result.abundance_maps[i, :, :])
        print(f"  {result.endmember_names[i]}: {mean_ab:.4f}")

    return result


def test_memory_manager():
    """显存管理器测试"""
    print("\n" + "=" * 60)
    print("测试 2: 显存管理器")
    print("=" * 60)

    import torch
    from src.memory_manager import MemoryManager, MemoryConfig

    config = MemoryConfig(
        device=DeviceType.CPU,
        max_batch_size=2048,
        min_batch_size=64,
    )

    mm = MemoryManager(config)
    print(f"设备: {mm.get_device()}")
    print(f"是否 CUDA: {mm.is_cuda()}")

    sample_bytes = 200 * 4
    batch_size = mm.optimize_batch_size(sample_bytes)
    print(f"优化后批大小: {batch_size}")

    tensor = mm.allocate_tensor((100, 200), dtype=torch.float32)
    print(f"张量形状: {tensor.shape}")
    print(f"张量设备: {tensor.device}")

    mm.empty_cache()
    print("显存清理完成")

    return True


def test_model_architecture():
    """模型架构测试"""
    print("\n" + "=" * 60)
    print("测试 3: 1D-CNN 自编码器模型")
    print("=" * 60)

    import torch
    from src.autoencoder import Autoencoder1D, Encoder1D, Decoder1D, LinearDecoder

    in_bands = 100
    num_endmembers = 4

    model = Autoencoder1D(
        in_bands=in_bands,
        num_endmembers=num_endmembers,
        hidden_dims=[32, 16],
        decoder_type="linear",
    )

    print(f"输入波段数: {in_bands}")
    print(f"端元数量: {num_endmembers}")
    print(f"总参数量: {model.count_parameters():,}")

    x = torch.randn(16, in_bands)
    recon, abundances = model(x)

    print(f"输入形状: {x.shape}")
    print(f"重建输出形状: {recon.shape}")
    print(f"丰度输出形状: {abundances.shape}")
    print(f"丰度和: {abundances.sum(dim=1)[:4]}")
    print(f"丰度非负: {(abundances >= 0).all().item()}")

    endmembers = model.get_endmembers()
    if endmembers is not None:
        print(f"端元矩阵形状: {endmembers.shape}")

    print("模型前向传播测试通过!")

    return True


def main():
    """运行所有测试"""
    import torch

    print("\n" + "=" * 60)
    print("高光谱解编推理引擎 - 综合测试")
    print("=" * 60)

    try:
        test_model_architecture()
    except Exception as e:
        print(f"模型架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        test_memory_manager()
    except Exception as e:
        print(f"显存管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        test_basic_unmixing()
    except Exception as e:
        print(f"基础解编测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
